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


# ---- repo_python is restricted to the reviewed patcher dir (optima/patchers/) ----


def _fake_repo(tmp_path):
    """A fake repo root with an optima/patchers/ dir and a stray non-patcher module."""
    repo = tmp_path / "repo"
    (repo / "optima" / "patchers").mkdir(parents=True)
    (repo / "optima" / "cli.py").write_text(  # a non-patcher repo module w/ a side effect
        f"open({str(tmp_path / 'cli_ran')!r}, 'w').close()\n"
    )
    return repo


def test_reviewed_patcher_under_patchers_dir_runs(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)
    marker = tmp_path / "patched"
    (repo / "optima" / "patchers" / "good.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    monkeypatch.setenv("OPTIMA_REPO_ROOT", str(repo))
    bundle = _bundle(tmp_path, {"steps": [{"type": "repo_python", "path": "good.py"}]})
    assert apply_rebuild_plan(bundle) is True
    assert marker.exists()  # the reviewed patcher ran


def test_repo_python_cannot_run_a_non_patcher_repo_module(tmp_path, monkeypatch):
    # The old behavior (containment-only) would have let this runpy optima/cli.py as __main__.
    repo = _fake_repo(tmp_path)
    monkeypatch.setenv("OPTIMA_REPO_ROOT", str(repo))
    bundle = _bundle(tmp_path, {"steps": [{"type": "repo_python", "path": "optima/cli.py"}]})
    with pytest.raises(RebuildError, match="escape|patcher"):
        apply_rebuild_plan(bundle)
    assert not (tmp_path / "cli_ran").exists()  # the stray module never executed


def test_repo_python_rejects_parent_traversal(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)
    monkeypatch.setenv("OPTIMA_REPO_ROOT", str(repo))
    bundle = _bundle(tmp_path, {"steps": [{"type": "repo_python", "path": "../cli.py"}]})
    with pytest.raises(RebuildError, match="simple relative|escape"):
        apply_rebuild_plan(bundle)
    assert not (tmp_path / "cli_ran").exists()


def test_repo_python_missing_patcher_fails_closed(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)
    monkeypatch.setenv("OPTIMA_REPO_ROOT", str(repo))
    bundle = _bundle(tmp_path, {"steps": [{"type": "repo_python", "path": "nope.py"}]})
    with pytest.raises(RebuildError, match="not found"):
        apply_rebuild_plan(bundle)


def test_repo_python_rejects_sibling_dir_that_prefixes_patchers(tmp_path, monkeypatch):
    # A sibling dir whose name STRING-prefixes "optima/patchers" (optima/patchers_evil)
    # takes the repo-relative branch, so containment (not the branch) must reject it.
    repo = _fake_repo(tmp_path)
    (repo / "optima" / "patchers_evil").mkdir()
    (repo / "optima" / "patchers_evil" / "x.py").write_text(
        f"open({str(tmp_path / 'evil_ran')!r}, 'w').close()\n"
    )
    monkeypatch.setenv("OPTIMA_REPO_ROOT", str(repo))
    bundle = _bundle(tmp_path, {"steps": [{"type": "repo_python", "path": "optima/patchers_evil/x.py"}]})
    with pytest.raises(RebuildError, match="escape|not found"):
        apply_rebuild_plan(bundle)
    assert not (tmp_path / "evil_ran").exists()


def test_repo_python_rejects_intermediate_symlinked_dir(tmp_path, monkeypatch):
    # A symlinked SUBDIR under patchers/ (not the leaf) that points outside the reviewed
    # set: the leaf isn't a symlink, so only resolve()+containment catches the escape.
    repo = _fake_repo(tmp_path)
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (outside / "x.py").write_text(f"open({str(tmp_path / 'outside_ran')!r}, 'w').close()\n")
    try:
        (repo / "optima" / "patchers" / "sub").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("no symlink support")
    monkeypatch.setenv("OPTIMA_REPO_ROOT", str(repo))
    bundle = _bundle(tmp_path, {"steps": [{"type": "repo_python", "path": "sub/x.py"}]})
    with pytest.raises(RebuildError, match="escape"):
        apply_rebuild_plan(bundle)
    assert not (tmp_path / "outside_ran").exists()


def test_repo_python_rejects_symlinked_patcher(tmp_path, monkeypatch):
    # A symlink inside the patcher dir could re-point at an unreviewed target.
    repo = _fake_repo(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text(f"open({str(tmp_path / 'outside_ran')!r}, 'w').close()\n")
    (repo / "optima" / "patchers" / "link.py").symlink_to(outside)
    monkeypatch.setenv("OPTIMA_REPO_ROOT", str(repo))
    bundle = _bundle(tmp_path, {"steps": [{"type": "repo_python", "path": "link.py"}]})
    with pytest.raises(RebuildError, match="symlink"):
        apply_rebuild_plan(bundle)
    assert not (tmp_path / "outside_ran").exists()
