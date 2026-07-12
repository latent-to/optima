"""SGLang plugin entry point (for sglang versions that ship the plugin framework).

SGLang's newer ``sglang.srt.plugins`` framework calls registered entry points
early in every process — including the spawned scheduler that runs the model.
Where that framework exists, this entry point installs the Optima seam there.

NOTE: not all released sglang versions have the plugin framework (e.g. 0.5.9
does not). The version-independent path is ``optima/bootstrap.py`` installed via
a ``.pth`` file; both funnel into ``optima.seam.activate``. Select this plugin
with ``SGLANG_PLUGINS=optima`` on versions that support it.
"""

from __future__ import annotations


def register() -> None:
    """Entry-point target invoked by sglang.load_plugins() in each process."""
    from optima import bootstrap, seam

    bootstrap.install()
    seam.activate()
