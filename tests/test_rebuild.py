"""The rebuild escape hatch must NEVER run bundle-supplied code (arbitrary RCE).

Only validator-shipped, reviewed ``repo_python`` patchers are allowed; a bundle that
tries to run its own script (`bundle_python`) is rejected *before* the script executes.
See docs/SLOT_CONTRACT.md. Pure-stdlib (no torch) — runs anywhere.
"""

from __future__ import annotations

import json

import pytest

from optima.rebuild import RebuildError, apply_rebuild_plan


def _bundle(tmp_path, plan):
    (tmp_path / "rebuild.json").write_text(json.dumps(plan))
    return tmp_path


def test_no_plan_is_noop(tmp_path):
    assert apply_rebuild_plan(tmp_path) is False  # no rebuild.json -> nothing to do


def test_bundle_python_is_rejected_without_executing(tmp_path):
    # A bundle trying to run its own code must be refused before the file runs.
    marker = tmp_path / "pwned"
    (tmp_path / "evil.py").write_text(f"open({str(marker)!r}, 'w').close()")
    bundle = _bundle(tmp_path, {"steps": [{"type": "bundle_python", "path": "evil.py"}]})
    with pytest.raises(RebuildError, match="not allowed"):
        apply_rebuild_plan(bundle)
    assert not marker.exists(), "bundle_python script executed — the fence is broken"


def test_unknown_step_is_rejected(tmp_path):
    bundle = _bundle(tmp_path, {"steps": [{"type": "curl_and_run", "path": "x"}]})
    with pytest.raises(RebuildError, match="unsupported"):
        apply_rebuild_plan(bundle)
