"""Reviewed rebuild patcher: compile a bundle's DECLARED CUDA sources, pre-load them.

This is the ONLY sanctioned compile path for the "CUDA source" bundle tier:

* The bundle ships INSPECTABLE ``.cu``/``.cuh`` source, declared per-op in its manifest
  (``ops[].cuda_sources``). ``scan_tree`` fails closed on undeclared non-``.py`` files,
  and the declared sources are folded into the per-slot copy fingerprints — the source
  is identity-hashed, copy-detected, and human-reviewable, unlike an opaque ``.so``
  (which the scanner rejects unconditionally, even if declared).
* This patcher is validator-shipped and reviewed (it lives under ``optima/patchers/``;
  ``rebuild.json`` may only select scripts here — see ``optima/rebuild.py``). It never
  executes bundle Python: it runs ``nvcc`` on the declared sources, nothing else.
* Artifacts are built OUTSIDE the bundle tree (a cache dir), so the built ``.so`` never
  enters the bundle's content hash or the scanned tree. Each compiled module is
  registered in ``sys.modules`` under its source stem (``fused_epilogue_sm103.cu`` ->
  ``import fused_epilogue_sm103``), so the bundle's shim reaches it with a PLAIN import
  — miner code needs no filesystem or import machinery (both sandbox-banned).
* Runs inside the isolated candidate process: ``apply_rebuild_plan`` is invoked by
  ``prepare_candidate_environment`` AFTER network isolation. The CLI never imports the
  result; only the spawned engine worker does.

CPU / dry-run boxes (no CUDA device or no ``nvcc``): skip with a notice — the bundle
shim falls back to its torch reference path (the contract exerciser, not the win).
A GPU box where the build FAILS raises: a silent skip there would score the fallback
path while looking like the kernel ran (a phantom-parity trap).

Compile recipe mirrors the campaign-proven ``build.sh`` (plain nvcc -> torch extension;
``-arch`` derived from the live device's compute capability, e.g. (10,3) -> sm_103a).
The source must ``#define``-free rely on ``TORCH_EXTENSION_NAME`` being injected: we
pass ``-DTORCH_EXTENSION_NAME=<stem>``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def _log(msg: str) -> None:
    print(f"[optima.build_cuda_ext] {msg}", flush=True)


def _cache_dir(bundle_id: str) -> Path:
    root = os.environ.get("OPTIMA_CUDA_EXT_CACHE", "")
    base = Path(root) if root else Path.home() / ".cache" / "optima" / "cuda_ext"
    return base / bundle_id


def _needs_build(src: Path, so: Path, stamp: Path) -> bool:
    if not so.is_file() or not stamp.is_file():
        return True
    return stamp.read_text().strip() != hashlib.sha256(src.read_bytes()).hexdigest()


def _compile(src: Path, so: Path, arch: str) -> None:
    import torch

    name = src.stem
    py_inc = sysconfig.get_paths()["include"]
    torch_root = Path(torch.__file__).parent
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    cmd = [
        "nvcc", "-O3", "--use_fast_math", "--std=c++17", f"-arch={arch}",
        "-Xcompiler", "-fPIC", "-shared",
        f"-DTORCH_EXTENSION_NAME={name}",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        f"-I{torch_root / 'include'}",
        f"-I{torch_root / 'include' / 'torch' / 'csrc' / 'api' / 'include'}",
        f"-I{py_inc}",
        str(src),
        f"-L{torch_root / 'lib'}",
        "-ltorch", "-ltorch_python", "-lc10", "-lc10_cuda", "-ltorch_cuda",
        "-o", str(so),
    ]
    _log(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _load(name: str, so: Path) -> None:
    import torch  # noqa: F401  — the extension links against libtorch; import it first

    spec = importlib.util.spec_from_file_location(name, so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    _log(f"loaded {name} ({so})")


def main() -> None:
    bundle = os.environ.get("OPTIMA_BUNDLE_PATH", "").strip()
    if not bundle:
        _log("no OPTIMA_BUNDLE_PATH set; nothing to build")
        return

    from optima.manifest import all_declared_cuda_sources, load_manifest

    manifest = load_manifest(bundle)
    sources = sorted(all_declared_cuda_sources(bundle, manifest))
    sources = [s for s in sources if Path(s).suffix == ".cu"]  # .cuh are headers, not units
    if not sources:
        _log("bundle declares no .cu compilation units; nothing to build")
        return

    import torch

    if not torch.cuda.is_available() or shutil.which("nvcc") is None:
        _log("no CUDA device or no nvcc on this box; SKIPPING build "
             "(the bundle shim falls back to its torch reference path)")
        return

    cc = torch.cuda.get_device_capability(0)
    arch = f"sm_{cc[0]}{cc[1]}a"
    cache = _cache_dir(manifest.bundle_id)
    cache.mkdir(parents=True, exist_ok=True)

    import fcntl

    for src_str in sources:
        src = Path(src_str)
        name = src.stem
        so = cache / f"{name}.so"
        stamp = cache / f"{name}.sha256"
        # Cross-PROCESS lock: engine TP ranks (and verify ranks without a barrier) all
        # run the rebuild plan concurrently; an unlocked build races nvcc's non-atomic
        # .so write against another rank's dlopen — the exact MSA JIT failure class
        # fixed on the pod this session. One rank builds, the rest wait, all load.
        with open(cache / f"{name}.lock", "w") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            if _needs_build(src, so, stamp):
                _compile(src, so, arch)  # raises on failure — no silent phantom-parity skip
                stamp.write_text(hashlib.sha256(src.read_bytes()).hexdigest())
            else:
                _log(f"cache hit for {name}")
        _load(name, so)


main()
