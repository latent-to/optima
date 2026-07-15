"""Bind FlashInfer to an exact, prebuilt dependency-source overlay.

The build worker materializes and the trusted controller publishes overlay bytes.
Engine ranks only validate and consume their read-only mount.  A declared overlay is
mandatory: missing, stale, writable, or corrupt state is terminal rather than a
warning followed by stock execution.
"""

from __future__ import annotations

import importlib
import logging
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("optima.flashinfer_overlay")

_installed = False


def _flashinfer_import_in_progress() -> bool:
    """Keep overlay installation out of FlashInfer's partial-import window.

    The bootstrap runs after every watched module import.  A different watched
    module can finish while ``flashinfer.jit.cubin_loader`` is still importing;
    importing the overlay's generator at that point re-enters FlashInfer and
    creates a circular import.  Returning here is safe because seam activation
    is repeated, including once at the positively identified scheduler entry.
    No overlay state may be mutated before this check.
    """

    for name, module in tuple(sys.modules.items()):
        if name != "flashinfer" and not name.startswith("flashinfer."):
            continue
        spec = getattr(module, "__spec__", None)
        if spec is not None and bool(getattr(spec, "_initializing", False)):
            return True
    return False


@dataclass(frozen=True)
class _LoadOnlyJITSpec:
    """The only runtime operation is loading one already-validated shared object."""

    name: str
    path: Path
    sha256: str
    size: int

    def build_and_load(self):
        from optima.dep_policy import sha256_file

        info = self.path.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size != self.size
            or info.st_mode & 0o222
            or sha256_file(self.path) != self.sha256
        ):
            raise RuntimeError(f"prebuilt dependency module changed before load: {self.path}")
        # Imported only inside the untrusted engine rank, never by the controller or
        # build worker.  No Ninja, lock, repair, or compilation path exists here.
        import tvm_ffi

        return tvm_ffi.load_module(str(self.path))


def _load_only_generator(module):
    def generate(use_fast_build: bool = False):
        if type(use_fast_build) is not bool or use_fast_build:
            raise RuntimeError(
                f"prebuilt dependency module {module.policy.name!r} supports only "
                "use_fast_build=False"
            )
        return _LoadOnlyJITSpec(
            module.policy.name, module.path, module.sha256, module.size
        )

    return generate


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _active_overlays():
    bundle = os.environ.get("OPTIMA_BUNDLE_PATH", "").strip()
    if not bundle or not _truthy(os.environ.get("OPTIMA_ACTIVE")):
        return []

    from optima.dep_policy import PATCHABLE_DEPS, read_validated_overlay
    from optima.manifest import load_manifest

    # Manifest parsing executes no bundle Python.  An active malformed tree is not a
    # reason to continue with stock dependency bytes.
    manifest = load_manifest(bundle)
    overlays = []
    for target in sorted({patch.target for patch in manifest.dep_patches}):
        policy = PATCHABLE_DEPS.get(target)
        if policy is None:
            raise RuntimeError(f"active tree declares unapproved dep target {target!r}")
        overlays.append((read_validated_overlay(bundle, target), policy))
    return overlays


def install(registry) -> None:  # registry unused; signature shared by integrations
    global _installed
    if _installed:
        return
    overlays = _active_overlays()
    if not overlays:
        return
    # This is a deferral, not a best-effort fallback: the scheduler-entry
    # activation retries installation after the framework import graph settles.
    # Check before env/module/cache mutation so every attempt is atomic.
    if _flashinfer_import_in_progress():
        return

    from optima import receipts

    applied: list[str] = []
    module_names: list[str] = []
    build_digests: set[str] = set()
    for overlay, policy in overlays:
        if policy.env_rebind is not None:
            module_name, attribute = policy.env_rebind
            environment = importlib.import_module(module_name)
            setattr(environment, attribute, overlay.subtree)
            logger.info("optima: %s.%s -> %s", module_name, attribute, overlay.subtree)
        target_architecture = os.environ.get("OPTIMA_TARGET_GPU_ARCH", "").strip().lower()
        # The artifact carries one module per fleet architecture; install only
        # the device's module and refuse devices the policy does not cover.
        selected = tuple(
            module
            for module in overlay.modules
            if module.policy.target_architecture == target_architecture
        )
        if overlay.modules and not selected:
            raise RuntimeError(
                f"dependency overlay {overlay.target!r} has no prebuilt module for "
                f"device architecture {target_architecture!r}"
            )
        for module in selected:
            replacement = _load_only_generator(module)
            generator_module = importlib.import_module(module.policy.generator_module)
            setattr(generator_module, module.policy.generator_attr, replacement)
            consumer = sys.modules.get(module.policy.consumer_module)
            if consumer is not None:
                setattr(consumer, module.policy.consumer_attr, replacement)
            module_names.append(module.policy.name)
        applied.append(overlay.target)
        build_digests.add(overlay.build_spec_digest)

    if len(build_digests) != 1:
        raise RuntimeError("active dependency overlays do not share one native-build identity")
    # A previously cached module would silently retain the stock AOT object.
    for _overlay, policy in overlays:
        for module in policy.prebuilt_modules:
            consumer = sys.modules.get(module.consumer_module)
            getter = getattr(consumer, "get_cutlass_fused_moe_module", None)
            if hasattr(getter, "cache_clear"):
                getter.cache_clear()

    build_spec_digest = next(iter(build_digests))
    _installed = True
    payload = {
        "targets": applied,
        "prebuilt_modules": sorted(module_names),
        "build_spec_digest": build_spec_digest,
    }
    receipts.write("overlay", payload)
    print(
        f"[optima] dep overlay ACTIVE: targets={applied} "
        f"modules={sorted(module_names)} build={build_spec_digest[:12]}",
        flush=True,
    )
