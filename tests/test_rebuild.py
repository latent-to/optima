"""Strict, data-only rebuild authority and shared execution projection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from optima.rebuild import (
    RebuildError,
    RebuildPlan,
    _main,
    apply_rebuild_plan,
    parse_rebuild_plan,
)


def test_rebuild_plan_schema_version_is_type_exact():
    with pytest.raises(RebuildError, match="schema_version"):
        RebuildPlan(schema_version=True, steps=())


def _bundle(tmp_path: Path, plan: object) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "rebuild.json").write_text(json.dumps(plan))
    return tmp_path


def _fake_repo(tmp_path: Path, *, marker: Path | None = None) -> Path:
    repo = tmp_path / "repo"
    patchers = repo / "optima" / "patchers"
    patchers.mkdir(parents=True)
    for name, label in (
        ("apply_dep_patch.py", "patch"),
        ("build_cuda_ext.py", "build"),
        ("build_cute_cubin.py", "cute-cubin"),
    ):
        body = "# reviewed\n"
        if marker is not None:
            body += f"open({str(marker)!r}, 'a').write({label!r} + '\\n')\n"
        (patchers / name).write_text(body)
    return repo


def _step(name: str) -> dict[str, str]:
    return {"type": "repo_python", "path": name}


def test_no_plan_is_a_shared_noop(tmp_path):
    assert parse_rebuild_plan(tmp_path) is None
    assert apply_rebuild_plan(tmp_path) is False


def test_parse_is_pure_and_canonicalizes_registered_order(tmp_path, monkeypatch):
    marker = tmp_path / "ran"
    repo = _fake_repo(tmp_path, marker=marker)
    monkeypatch.setenv("OPTIMA_REPO_ROOT", str(repo))
    bundle = _bundle(
        tmp_path / "bundle",
        {
            "steps": [
                _step("build_cute_cubin.py"),
                _step("build_cuda_ext.py"),
                _step("apply_dep_patch.py"),
            ]
        },
    )

    plan = parse_rebuild_plan(bundle)
    assert plan is not None and not marker.exists()
    assert [step.patcher_id for step in plan.steps] == [
        "optima.apply-dep-patch.v1",
        "optima.build-cuda-ext.v1",
        "optima.build-cute-cubin.v1",
    ]
    assert plan.to_dict() == {
        "steps": [
            _step("optima/patchers/apply_dep_patch.py"),
            _step("optima/patchers/build_cuda_ext.py"),
            _step("optima/patchers/build_cute_cubin.py"),
        ]
    }
    assert all(len(step.patcher_sha256) == 64 for step in plan.steps)
    assert plan.identity_data()["steps"][0]["patcher_id"] == (
        "optima.apply-dep-patch.v1"
    )

    assert apply_rebuild_plan(bundle) is True
    assert marker.read_text().splitlines() == [
        "patch",
        "build",
        "cute-cubin",
    ]


@pytest.mark.parametrize("phase", ["all", "build", "load"])
def test_rebuild_phase_is_passed_exactly_and_environment_is_restored(
    tmp_path, monkeypatch, phase
):
    marker = tmp_path / "phase"
    repo = _fake_repo(tmp_path)
    patcher = repo / "optima" / "patchers" / "build_cuda_ext.py"
    patcher.write_text(
        "import os\n"
        f"open({str(marker)!r}, 'w').write(os.environ['OPTIMA_REBUILD_PHASE'])\n"
    )
    monkeypatch.setenv("OPTIMA_REPO_ROOT", str(repo))
    monkeypatch.setenv("OPTIMA_REBUILD_PHASE", "sentinel")
    bundle = _bundle(
        tmp_path / "bundle", {"steps": [_step("build_cuda_ext.py")]}
    )

    assert apply_rebuild_plan(bundle, phase=phase) is True
    assert marker.read_text() == phase
    assert __import__("os").environ["OPTIMA_REBUILD_PHASE"] == "sentinel"


def test_unknown_rebuild_phase_rejects_before_patcher_execution(tmp_path, monkeypatch):
    marker = tmp_path / "ran"
    monkeypatch.setenv("OPTIMA_REPO_ROOT", str(_fake_repo(tmp_path, marker=marker)))
    bundle = _bundle(
        tmp_path / "bundle", {"steps": [_step("build_cuda_ext.py")]}
    )
    with pytest.raises(RebuildError, match="unsupported rebuild phase"):
        apply_rebuild_plan(bundle, phase="candidate")  # type: ignore[arg-type]
    assert not marker.exists()


@pytest.mark.parametrize(
    "phase,message",
    [
        ("build", "disposable rebuild container"),
        ("load", "isolated engine worker"),
        ("all", "development-only"),
    ],
)
def test_module_entry_requires_phase_specific_container_authority(
    tmp_path, monkeypatch, phase, message
):
    monkeypatch.delenv("OPTIMA_REBUILD_CONTAINER", raising=False)
    monkeypatch.delenv("OPTIMA_ENGINE_WORKER", raising=False)
    monkeypatch.delenv("OPTIMA_REBUILD_DEVELOPMENT", raising=False)
    with pytest.raises(RebuildError, match=message):
        _main(["--phase", phase, str(tmp_path)])


@pytest.mark.parametrize(
    "phase,environment",
    [
        ("build", ("OPTIMA_REBUILD_CONTAINER", "1")),
        ("load", ("OPTIMA_ENGINE_WORKER", "1")),
        ("all", ("OPTIMA_REBUILD_DEVELOPMENT", "1")),
    ],
)
def test_module_entry_accepts_only_explicit_internal_or_development_lane(
    tmp_path, monkeypatch, phase, environment
):
    monkeypatch.setenv(*environment)
    assert _main(["--phase", phase, str(tmp_path)]) == 0


def test_parser_normalizes_the_only_two_accepted_path_spellings(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)
    monkeypatch.setenv("OPTIMA_REPO_ROOT", str(repo))
    short = parse_rebuild_plan(
        _bundle(tmp_path / "short", {"steps": [_step("build_cuda_ext.py")]})
    )
    full = parse_rebuild_plan(
        _bundle(
            tmp_path / "full",
            {"steps": [_step("optima/patchers/build_cuda_ext.py")]},
        )
    )
    assert short is not None and full is not None
    assert short.identity_data() == full.identity_data()


@pytest.mark.parametrize(
    "plan, message",
    [
        ([], "must be an object"),
        ({}, "exactly.*steps"),
        ({"steps": [], "extra": 1}, "exactly.*steps"),
        ({"steps": {}}, "must be a list"),
        ({"steps": [7]}, "step 0 must be an object"),
        ({"steps": [{"type": "repo_python"}]}, "exactly.*type.*path"),
        (
            {"steps": [{"type": "repo_python", "path": "build_cuda_ext.py", "x": 1}]},
            "exactly.*type.*path",
        ),
        ({"steps": [_step("../build_cuda_ext.py")]}, "traversal"),
        ({"steps": [_step("optima/cli.py")]}, "registered file"),
        ({"steps": [_step("unreviewed.py")]}, "unregistered"),
        (
            {"steps": [{"type": "bundle_python", "path": "evil.py"}]},
            "not allowed",
        ),
        (
            {"steps": [{"type": "curl_and_run", "path": "x"}]},
            "unsupported",
        ),
        (
            {"steps": [_step("build_cuda_ext.py"), _step("build_cuda_ext.py")]},
            "duplicate rebuild patcher",
        ),
    ],
)
def test_strict_plan_schema_rejects_unregistered_authority(
    tmp_path, monkeypatch, plan, message
):
    monkeypatch.setenv("OPTIMA_REPO_ROOT", str(_fake_repo(tmp_path)))
    marker = tmp_path / "pwned"
    (tmp_path / "bundle" / "evil.py").parent.mkdir(parents=True)
    (tmp_path / "bundle" / "evil.py").write_text(
        f"open({str(marker)!r}, 'w').close()"
    )
    bundle = _bundle(tmp_path / "bundle", plan)
    with pytest.raises(RebuildError, match=message):
        apply_rebuild_plan(bundle)
    assert not marker.exists()


def test_duplicate_json_keys_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIMA_REPO_ROOT", str(_fake_repo(tmp_path)))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "rebuild.json").write_text('{"steps": [], "steps": []}')
    with pytest.raises(RebuildError, match="duplicate key"):
        parse_rebuild_plan(bundle)


def test_registered_patcher_must_exist_and_be_regular(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)
    (repo / "optima" / "patchers" / "build_cuda_ext.py").unlink()
    monkeypatch.setenv("OPTIMA_REPO_ROOT", str(repo))
    bundle = _bundle(
        tmp_path / "bundle", {"steps": [_step("build_cuda_ext.py")]}
    )
    with pytest.raises(RebuildError, match="not found"):
        parse_rebuild_plan(bundle)


def test_registered_patcher_symlink_is_rejected(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)
    patcher = repo / "optima" / "patchers" / "build_cuda_ext.py"
    outside = tmp_path / "outside.py"
    outside.write_text("raise AssertionError('must not run')\n")
    patcher.unlink()
    try:
        patcher.symlink_to(outside)
    except OSError:
        pytest.skip("no symlink support")
    monkeypatch.setenv("OPTIMA_REPO_ROOT", str(repo))
    bundle = _bundle(
        tmp_path / "bundle", {"steps": [_step("build_cuda_ext.py")]}
    )
    with pytest.raises(RebuildError, match="symlink"):
        apply_rebuild_plan(bundle)


def test_dangling_rebuild_plan_symlink_is_rejected(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    try:
        (bundle / "rebuild.json").symlink_to(bundle / "missing.json")
    except OSError:
        pytest.skip("no symlink support")

    with pytest.raises(RebuildError, match="non-symlink"):
        parse_rebuild_plan(bundle)


def test_patcher_source_bytes_are_part_of_identity(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)
    monkeypatch.setenv("OPTIMA_REPO_ROOT", str(repo))
    bundle = _bundle(
        tmp_path / "bundle", {"steps": [_step("build_cuda_ext.py")]}
    )
    before = parse_rebuild_plan(bundle)
    assert before is not None
    (repo / "optima" / "patchers" / "build_cuda_ext.py").write_text(
        "# reviewed revision two\n"
    )
    after = parse_rebuild_plan(bundle)
    assert after is not None
    assert before.identity_data() != after.identity_data()
