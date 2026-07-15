import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

import pytest

from optima import receipts, seam
from optima.eval import _launch, engine_worker


def _sandbox_proc_reader(*, seccomp: int = 2, filters: int = 1, caps: int = 0):
    values = {
        "/proc/self/status": (
            f"CapEff:\t{caps:x}\n"
            f"CapBnd:\t{caps:x}\n"
            "NoNewPrivs:\t1\n"
            f"Seccomp:\t{seccomp}\n"
            f"Seccomp_filters:\t{filters}\n"
        ),
        "/proc/sys/kernel/yama/ptrace_scope": "1\n",
        "/proc/mounts": "overlay / overlay ro,relatime 0 0\n",
    }

    def read_text(path: Path, *args, **kwargs):
        return values[str(path)]

    return read_text


class IsolationTests(TestCase):
    def test_sandbox_probes_are_shared_engine_worker_compatibility_aliases(self):
        for name in (
            "_truthy_env",
            "_loopback_is_up",
            "_network_namespace_is_loopback_only",
            "_egress_is_blocked",
            "_process_sandbox_is_hardened",
            "_path_mount_is_read_only",
        ):
            self.assertIs(getattr(_launch, name), getattr(engine_worker, name))
            self.assertEqual(getattr(_launch, name).__module__, engine_worker.__name__)

    def test_process_hardening_requires_zero_caps_and_live_seccomp(self):
        with mock.patch.object(
            Path, "read_text", autospec=True, side_effect=_sandbox_proc_reader()
        ):
            self.assertTrue(_launch._process_sandbox_is_hardened())
        for kwargs in (
            {"seccomp": 0},
            {"filters": 0},
            {"caps": 1 << 23},
        ):
            with mock.patch.object(
                Path,
                "read_text",
                autospec=True,
                side_effect=_sandbox_proc_reader(**kwargs),
            ):
                self.assertFalse(_launch._process_sandbox_is_hardened())

    def test_external_isolation_is_live_verified(self):
        with mock.patch.dict("os.environ", {"OPTIMA_EXTERNAL_NO_EGRESS": "1"}), \
             mock.patch.object(_launch, "_loopback_is_up", return_value=True), \
             mock.patch.object(
                 _launch, "_network_namespace_is_loopback_only", return_value=True
             ), \
             mock.patch.object(_launch, "_egress_is_blocked", return_value=True), \
             mock.patch.object(
                 _launch, "_process_sandbox_is_hardened", return_value=True
             ):
            self.assertTrue(_launch.isolate_network())

    def test_external_isolation_claim_fails_any_live_check(self):
        checks = (
            "_loopback_is_up",
            "_network_namespace_is_loopback_only",
            "_egress_is_blocked",
            "_process_sandbox_is_hardened",
        )
        for failed in checks:
            patches = {
                name: mock.patch.object(_launch, name, return_value=name != failed)
                for name in checks
            }
            with mock.patch.dict(
                "os.environ", {"OPTIMA_EXTERNAL_NO_EGRESS": "1"}
            ), patches[checks[0]], patches[checks[1]], patches[checks[2]], patches[checks[3]]:
                self.assertFalse(_launch.isolate_network())

    def test_requested_isolation_fails_closed(self):
        cfg = SimpleNamespace(
            isolate=True,
            framework_mode=False,
            allow_unsafe_no_isolation=False,
        )

        with mock.patch.object(_launch, "isolate_network", return_value=False):
            with self.assertRaisesRegex(_launch.IsolationError, "could not be proven"):
                _launch.prepare_candidate_environment(cfg, bundle_path="", active=True)

    def test_framework_mode_requires_isolation_by_default(self):
        cfg = SimpleNamespace(
            isolate=False,
            framework_mode=True,
            allow_unsafe_no_isolation=False,
        )

        with self.assertRaisesRegex(_launch.IsolationError, "framework_mode requires"):
            _launch.prepare_candidate_environment(cfg, bundle_path="", active=True)

    def test_unsafe_dev_override_allows_failed_isolation(self):
        cfg = SimpleNamespace(
            isolate=True,
            framework_mode=True,
            allow_unsafe_no_isolation=True,
        )

        with mock.patch.object(_launch, "isolate_network", return_value=False):
            _launch.prepare_candidate_environment(cfg, bundle_path="", active=True)


def test_engine_kwargs_preserve_candidate_overrides_and_legacy_alias():
    assert _launch.engine_kwargs is engine_worker.engine_kwargs
    cfg = SimpleNamespace(
        model_path="model",
        dtype="float16",
        mem_fraction_static=0.8,
        seed=7,
        log_level="error",
        attention_backend="baseline-attention",
        candidate_attention_backend="candidate-attention",
        moe_runner_backend="baseline-moe",
        candidate_moe_runner_backend="candidate-moe",
        disable_custom_all_reduce=False,
        candidate_disable_custom_all_reduce=True,
        extra_engine_kwargs={"page_size": 32},
        candidate_extra_engine_kwargs={"page_size": 64},
    )
    baseline = engine_worker.engine_kwargs(cfg, active=False)
    candidate = engine_worker.engine_kwargs(cfg, active=True)
    assert baseline["attention_backend"] == "baseline-attention"
    assert baseline["moe_runner_backend"] == "baseline-moe"
    assert baseline["page_size"] == 32
    assert "disable_custom_all_reduce" not in baseline
    assert candidate["attention_backend"] == "candidate-attention"
    assert candidate["moe_runner_backend"] == "candidate-moe"
    assert candidate["disable_custom_all_reduce"] is True
    assert candidate["page_size"] == 64


def test_direct_native_target_architecture_covers_exact_visible_tp(
    monkeypatch, tmp_path
):
    (tmp_path / "rebuild.json").write_text('{"steps": []}')
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.delenv("OPTIMA_TARGET_GPU_ARCH", raising=False)
    monkeypatch.delenv("OPTIMA_ENGINE_WORKER", raising=False)
    monkeypatch.delenv("OPTIMA_PREBUILT_ARTIFACTS", raising=False)
    observed = []

    def run(command, **kwargs):
        observed.append((command, kwargs))
        return SimpleNamespace(stdout="10.3\n10.3\n10.3\n10.3\n")

    monkeypatch.setattr(_launch.subprocess, "run", run)
    cfg = SimpleNamespace(tp_size=4)
    assert _launch._direct_native_target_architecture(
        cfg, bundle_path=str(tmp_path), active=True
    ) == "sm103"
    assert observed[0][0][-2:] == ["-i", "4,5,6,7"]
    assert observed[0][1]["check"] is True


def test_direct_native_target_architecture_rejects_mixed_or_stale_claim(
    monkeypatch, tmp_path
):
    (tmp_path / "rebuild.json").write_text('{"steps": []}')
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.delenv("OPTIMA_ENGINE_WORKER", raising=False)
    monkeypatch.delenv("OPTIMA_PREBUILT_ARTIFACTS", raising=False)
    cfg = SimpleNamespace(tp_size=2)

    monkeypatch.setattr(
        _launch.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="10.3\n10.0\n"),
    )
    with pytest.raises(_launch.IsolationError, match="homogeneous"):
        _launch._direct_native_target_architecture(
            cfg, bundle_path=str(tmp_path), active=True
        )

    monkeypatch.setenv("OPTIMA_TARGET_GPU_ARCH", "sm120")
    monkeypatch.setattr(
        _launch.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="10.3\n10.3\n"),
    )
    with pytest.raises(_launch.IsolationError, match="differs from live"):
        _launch._direct_native_target_architecture(
            cfg, bundle_path=str(tmp_path), active=True
        )


def test_prepare_and_entry_share_one_module_instance(tmp_path):
    # A (prepare, forward) op's callables must come from ONE module execution: the
    # seam/verify loaders pull both off a single load_module. Two load_entry calls
    # would re-run the body (side effects twice) and split module globals so state
    # written by prepare would be invisible to entry.
    src = tmp_path / "k.py"
    src.write_text(
        "COUNT = 0\n"
        "_STATE = {}\n"
        "def prepare(w13, w2):\n"
        "    _STATE['p'] = 1\n"
        "    return (w13, w2)\n"
        "def entry(*args):\n"
        "    return _STATE.get('p')\n"
    )
    from optima.sandbox import callable_from, load_module

    module = load_module(src)
    prepare = callable_from(module, "prepare")
    entry = callable_from(module, "entry")
    prepare(None, None)
    assert entry() == 1  # shared globals: entry sees what prepare wrote

    # and the documented hazard is real: a SECOND load is a fresh namespace
    from optima.sandbox import load_entry

    entry2 = load_entry(src, "entry")
    assert entry2() is None


def _setup_bundle(tmp_path, *, include_setup=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "kernel.py").write_text(
        "def entry(x, out):\n    out.copy_(x)\n"
        "def patch_engine():\n    return None\n"
    )
    setup_line = 'setup = "patch_engine"\n' if include_setup else ""
    (tmp_path / "manifest.toml").write_text(
        'bundle_id = "setup-test"\n'
        'abi_version = "optima-op-abi-v0"\n'
        "[[ops]]\n"
        'slot = "activation.silu_and_mul"\n'
        'source = "kernel.py"\n'
        'entry = "entry"\n'
        + setup_line
    )
    return tmp_path


def _isolation_cfg(*, framework_mode, isolate=True, allow_unsafe=False):
    return SimpleNamespace(
        isolate=isolate,
        framework_mode=framework_mode,
        allow_unsafe_no_isolation=allow_unsafe,
    )


def test_setup_bundle_requires_explicit_framework_mode_before_isolation(tmp_path):
    bundle = _setup_bundle(tmp_path)
    with mock.patch.object(_launch, "isolate_network", return_value=True) as isolate:
        with pytest.raises(_launch.IsolationError, match="declares setup"):
            _launch.prepare_candidate_environment(
                _isolation_cfg(framework_mode=False),
                bundle_path=str(bundle),
                active=True,
            )
    isolate.assert_not_called()


def test_armed_setup_and_ordinary_bundle_pass_candidate_preflight(tmp_path):
    setup_bundle = _setup_bundle(tmp_path / "setup")
    ordinary_bundle = _setup_bundle(tmp_path / "ordinary", include_setup=False)
    with mock.patch.object(_launch, "isolate_network", return_value=True):
        _launch.prepare_candidate_environment(
            _isolation_cfg(framework_mode=True),
            bundle_path=str(setup_bundle),
            active=True,
        )
        _launch.prepare_candidate_environment(
            _isolation_cfg(framework_mode=False),
            bundle_path=str(ordinary_bundle),
            active=True,
        )


def test_setup_cannot_use_unsafe_no_isolation_override(tmp_path):
    bundle = _setup_bundle(tmp_path)
    with pytest.raises(_launch.IsolationError, match="unsafe development override"):
        _launch.prepare_candidate_environment(
            _isolation_cfg(
                framework_mode=True, isolate=False, allow_unsafe=True
            ),
            bundle_path=str(bundle),
            active=True,
        )


def test_setup_cannot_bypass_failed_isolation(tmp_path):
    bundle = _setup_bundle(tmp_path)
    with mock.patch.object(_launch, "isolate_network", return_value=False):
        with pytest.raises(_launch.IsolationError, match="failed fence"):
            _launch.prepare_candidate_environment(
                _isolation_cfg(framework_mode=True, allow_unsafe=True),
                bundle_path=str(bundle),
                active=True,
            )


def test_external_worker_requires_read_only_inputs_and_prebuild_binding(
    tmp_path, monkeypatch
):
    bundle = _setup_bundle(tmp_path)
    cfg = _isolation_cfg(framework_mode=True)
    cfg.model_path = "/models/model"
    external = {
        "OPTIMA_EXTERNAL_NO_EGRESS": "1",
        "OPTIMA_ENGINE_WORKER": "1",
        "OPTIMA_PREBUILT_ARTIFACTS": "1",
        "OPTIMA_NATIVE_BUILD_SPEC_DIGEST": "a" * 64,
        "OPTIMA_NATIVE_ARTIFACT_ROOT": "/optima/native",
        "OPTIMA_NATIVE_ARTIFACT_PUBLICATION_DIGEST": "b" * 64,
    }
    monkeypatch.setattr(_launch, "isolate_network", lambda: True)
    monkeypatch.setattr(_launch, "_dep_overlay_env", lambda _bundle: None)
    monkeypatch.setenv("OPTIMA_EXTERNAL_NO_EGRESS", "1")
    monkeypatch.setenv("OPTIMA_ENGINE_WORKER", "1")
    monkeypatch.setattr(_launch, "_path_mount_is_read_only", lambda _path: False)
    with pytest.raises(_launch.IsolationError, match="mounted read-only"):
        _launch.prepare_candidate_environment(
            cfg, bundle_path=str(bundle), active=True
        )

    monkeypatch.setattr(_launch, "_path_mount_is_read_only", lambda _path: True)
    with pytest.raises(_launch.IsolationError, match="prebuild binding"):
        _launch.prepare_candidate_environment(
            cfg, bundle_path=str(bundle), active=True
        )

    for name, value in external.items():
        monkeypatch.setenv(name, value)
    with mock.patch("optima.rebuild.apply_rebuild_plan") as rebuild:
        _launch.prepare_candidate_environment(
            cfg, bundle_path=str(bundle), active=True
        )
    rebuild.assert_not_called()


@pytest.mark.parametrize(
    ("active", "framework_mode", "expected"),
    ((False, True, "0"), (True, False, "0"), (True, True, "1")),
)
def test_launch_scopes_framework_arming_and_restores_parent(
    monkeypatch, active, framework_mode, expected
):
    observed = []

    class Engine:
        def __init__(self, **_kwargs):
            observed.append(os.environ.get("OPTIMA_FRAMEWORK_MODE"))

        def shutdown(self):
            return None

    cfg = SimpleNamespace(
        model_path="model",
        dtype="float32",
        mem_fraction_static=0.1,
        seed=0,
        log_level="error",
        framework_mode=framework_mode,
    )
    monkeypatch.setenv("OPTIMA_FRAMEWORK_MODE", "ambient")
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=Engine))
    monkeypatch.setattr(seam, "mark_driver", lambda: None)
    monkeypatch.setattr(_launch, "prepare_candidate_environment", lambda *_a, **_k: None)
    monkeypatch.setattr(_launch, "_wait_gpu_drain", lambda: None)
    monkeypatch.setattr(
        receipts,
        "require",
        lambda *_a, **_k: [
            {"slots": ["s"], "pid": 10, "rank": -1, "world_size": -1}
        ],
    )
    monkeypatch.setattr(
        _launch, "_require_execution_completion", lambda *_a, **_k: "ok"
    )

    with _launch.launched_engine(
        cfg, bundle_path="bundle" if active else "", active=active
    ):
        assert os.environ["OPTIMA_FRAMEWORK_MODE"] == expected

    assert observed == [expected]
    assert os.environ["OPTIMA_FRAMEWORK_MODE"] == "ambient"
