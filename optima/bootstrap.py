"""Version-independent seam loader via a post-import hook.

SGLang's released versions don't all ship the ``sglang.srt.plugins`` framework,
and ``sglang.Engine`` runs the model in a spawned scheduler process. To install
the Optima seam reliably in *every* interpreter in the venv — including that
spawned child — we drop a ``.pth`` file in site-packages containing the single
line ``import optima.bootstrap``. Python executes ``.pth`` imports at interpreter
startup (in spawned children too), so this module loads everywhere.

At startup sglang is not yet imported, and importing it here would be heavy and
fragile. So instead we register a meta-path finder that defers patching until
``sglang.srt.layers.activation`` is actually imported, then runs ``seam.activate``
against the freshly-loaded module. ``seam.activate`` is env-driven, so a baseline
process (OPTIMA_ACTIVE unset) just installs a pass-through dispatcher.
"""

from __future__ import annotations

import importlib.abc
import os
import sys

# Modules whose import should trigger seam installation — derived from the single seam
# table (optima/seams.py), so adding a seam there is the only edit. seams.py is stdlib-only
# (no torch/sglang), safe to import at interpreter startup. seam.activate() installs
# whatever is loaded.
from optima.seams import TARGET_MODULES as _TARGETS


# Install the standard-library-only discovery role hook before any SGLang
# import.  The hook is inert unless the isolated worker explicitly arms a
# sealed overlay, and only exact scheduler spawn children receive that overlay.
if os.environ.get("OPTIMA_DISCOVERY_OVERLAY_ARMED", "").strip().lower() in {
    "1", "true", "yes", "on",
}:
    from optima import discovery_overlay as _discovery_overlay

    _discovery_overlay.install()


def _run_activate() -> None:
    try:
        from optima import seam

        seam.activate()
    except Exception:  # noqa: BLE001 - never break the host interpreter
        import logging

        logging.getLogger("optima.bootstrap").exception("optima: seam activate failed")


def _wrap_loader(loader):
    orig_exec = loader.exec_module

    def exec_module(module):
        orig_exec(module)
        _run_activate()

    loader.exec_module = exec_module
    return loader


class _SeamFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS:
            return None
        # Resolve the real spec via the other finders, then wrap its loader so
        # our callback runs right after the module body executes.
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is None:
                continue
            if spec.loader is not None:
                spec.loader = _wrap_loader(spec.loader)
            return spec
        return None


def install() -> None:
    if any(t in sys.modules for t in _TARGETS):
        # Something already imported (e.g. re-entry); patch what's available now.
        _run_activate()
    if not any(isinstance(f, _SeamFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _SeamFinder())


install()
