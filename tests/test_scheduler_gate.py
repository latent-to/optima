"""The scheduler-role bundle-load gate (F4, B300 2026-07-13).

sglang's spawned detokenizer imports watched seam modules transitively, so an
import-time bundle load executes miner module-level code in an OUTPUT-PATH
process (the output-substitution surface) and over-counts the active-member
coverage gate (5/4 at TP4). The fix: ``seam.activate()`` arms but never loads;
``seam.load_candidate_bundle()`` loads only at ``run_scheduler_process`` entry
via the scheduler_gate adapter. These tests pin both halves.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from optima import receipts, seam
from optima.integrations import sglang_artifact_context as artifact_context
from optima.integrations import sglang_scheduler_gate as gate
from optima.registry import REGISTRY

SILU_BUNDLE = Path(__file__).parent.parent / "examples" / "miner_silu_torch"

_SCHED_MODULE = "sglang.srt.managers.scheduler"
_MODEL_RUNNER_MODULE = "sglang.srt.model_executor.model_runner"


@pytest.fixture()
def armed_env(tmp_path, monkeypatch):
    rdir = tmp_path / "receipts"
    monkeypatch.setenv("OPTIMA_SEAM_RECEIPT_DIR", str(rdir))
    monkeypatch.setenv("OPTIMA_ACTIVE", "1")
    monkeypatch.setenv("OPTIMA_BUNDLE_PATH", str(SILU_BUNDLE))
    monkeypatch.delenv("OPTIMA_RELEASE_REQUIRED", raising=False)
    monkeypatch.setattr(receipts, "_ONCE", set())
    monkeypatch.setattr(seam, "_bundle_loaded", False)
    monkeypatch.setattr(seam, "_bundle_pending", None)
    monkeypatch.setattr(seam, "_IS_DRIVER", False)
    yield rdir
    REGISTRY.clear()
    REGISTRY.disable()


@pytest.fixture()
def fake_scheduler_module(monkeypatch):
    mod = types.ModuleType(_SCHED_MODULE)
    calls: list[tuple[tuple, dict]] = []

    def run_scheduler_process(*args, **kwargs):
        calls.append((args, kwargs))
        return "scheduler-ran"

    mod.run_scheduler_process = run_scheduler_process
    mod._calls = calls
    monkeypatch.setitem(sys.modules, _SCHED_MODULE, mod)
    return mod


@pytest.fixture()
def fake_model_runner_module(monkeypatch):
    mod = types.ModuleType(_MODEL_RUNNER_MODULE)
    calls: list[str] = []

    class ModelRunner:
        def __init__(self, *, is_draft_worker=False):
            self.is_draft_worker = is_draft_worker

        def init_torch_distributed(self):
            calls.append("sglang-context")
            return "memory-snapshot"

    mod.ModelRunner = ModelRunner
    mod._calls = calls
    monkeypatch.setitem(sys.modules, _MODEL_RUNNER_MODULE, mod)
    return mod


def test_activate_arms_but_never_loads(armed_env):
    seam.activate()
    assert REGISTRY.slots() == []
    assert not seam._bundle_loaded
    # no positive evidence may exist before a scheduler proves its role
    assert not armed_env.exists() or receipts.collect(armed_env, "active") == []


def test_load_candidate_bundle_loads_and_receipts(armed_env):
    seam.load_candidate_bundle()
    assert seam._bundle_loaded
    assert "activation.silu_and_mul" in REGISTRY.slots()
    active = receipts.collect(armed_env, "active")
    assert len(active) == 1
    assert "activation.silu_and_mul" in active[0]["slots"]
    # idempotent: a second call must not double-register or double-receipt
    seam.load_candidate_bundle()
    assert len(receipts.collect(armed_env, "active")) == 1


def test_direct_bundle_stages_until_post_device_hook(
    armed_env, monkeypatch
):
    from optima import manifest
    from optima.registry import KernelImpl

    observed: list[str] = []
    monkeypatch.setattr(
        manifest,
        "load_manifest",
        lambda _bundle: types.SimpleNamespace(
            ops=(types.SimpleNamespace(aot_exports=(object(),)),)
        ),
    )

    def load_after_context(bundle):
        observed.append(bundle)
        REGISTRY.register(
            KernelImpl(
                slot="activation.silu_and_mul",
                bundle_id="direct-test",
                entry=lambda *_args: None,
            )
        )

    monkeypatch.setattr(seam, "_load_bundle_into_registry", load_after_context)
    monkeypatch.setattr(seam, "_install_adapters", lambda _required: None)

    seam.load_candidate_bundle()
    assert seam._bundle_pending == (str(SILU_BUNDLE), False)
    assert not seam._bundle_loaded
    assert REGISTRY.slots() == []
    assert observed == []
    assert not armed_env.exists() or receipts.collect(armed_env, "active") == []

    seam.finalize_pending_candidate_bundle()
    assert seam._bundle_pending is None
    assert seam._bundle_loaded
    assert REGISTRY.slots() == ["activation.silu_and_mul"]
    assert observed == [str(SILU_BUNDLE)]
    assert len(receipts.collect(armed_env, "active")) == 1

    # A draft/additional ModelRunner sees the same hook but cannot reload it.
    seam.finalize_pending_candidate_bundle()
    assert observed == [str(SILU_BUNDLE)]


def test_load_candidate_bundle_is_inert_in_the_driver(armed_env, monkeypatch):
    monkeypatch.setattr(seam, "_IS_DRIVER", True)
    seam.load_candidate_bundle()
    assert not seam._bundle_loaded
    assert REGISTRY.slots() == []
    assert not armed_env.exists() or receipts.collect(armed_env, "active") == []


def test_load_candidate_bundle_is_inert_when_unarmed(armed_env, monkeypatch):
    monkeypatch.setenv("OPTIMA_ACTIVE", "0")
    seam.load_candidate_bundle()
    assert not seam._bundle_loaded
    assert REGISTRY.slots() == []


def test_gate_wraps_scheduler_entry_and_loads(armed_env, fake_scheduler_module):
    gate.install()
    assert gate.is_installed()
    wrapped = fake_scheduler_module.run_scheduler_process
    gate.install()  # idempotent
    assert fake_scheduler_module.run_scheduler_process is wrapped

    result = fake_scheduler_module.run_scheduler_process(7, key="v")
    assert result == "scheduler-ran"
    assert fake_scheduler_module._calls == [((7,), {"key": "v"})]
    assert seam._bundle_loaded
    assert "activation.silu_and_mul" in REGISTRY.slots()
    assert len(receipts.collect(armed_env, "active")) == 1

    gate.uninstall()
    assert not gate.is_installed()


def test_gate_delegates_without_loading_when_unarmed(
    armed_env, fake_scheduler_module, monkeypatch
):
    monkeypatch.setenv("OPTIMA_ACTIVE", "0")
    gate.install()
    assert fake_scheduler_module.run_scheduler_process(1) == "scheduler-ran"
    assert not seam._bundle_loaded
    assert REGISTRY.slots() == []
    gate.uninstall()


def test_gate_install_noops_without_the_module(monkeypatch):
    monkeypatch.delitem(sys.modules, _SCHED_MODULE, raising=False)
    gate.install()  # must not raise
    assert not gate.is_installed()


def test_gate_tears_down_after_success(monkeypatch, fake_scheduler_module):
    observed: list[object] = []
    monkeypatch.setattr(seam, "load_candidate_bundle", lambda: observed.append("load"))
    monkeypatch.setattr(
        seam,
        "teardown_candidate_bundle",
        lambda *, suppress_errors=False: observed.append(("close", suppress_errors)),
    )
    gate.install()

    assert fake_scheduler_module.run_scheduler_process() == "scheduler-ran"
    assert observed == ["load", ("close", False)]


def test_gate_preserves_engine_error_and_suppresses_secondary_teardown(
    monkeypatch, fake_scheduler_module
):
    observed: list[object] = []

    def fail_engine(*_args, **_kwargs):
        raise RuntimeError("initiating rank failure")

    fake_scheduler_module.run_scheduler_process = fail_engine
    monkeypatch.setattr(seam, "load_candidate_bundle", lambda: observed.append("load"))

    def teardown(*, suppress_errors=False):
        observed.append(("close", suppress_errors))
        if not suppress_errors:
            raise RuntimeError("secondary teardown failure")

    monkeypatch.setattr(seam, "teardown_candidate_bundle", teardown)
    gate.install()

    with pytest.raises(RuntimeError, match="initiating rank failure"):
        fake_scheduler_module.run_scheduler_process()
    assert observed == ["load", ("close", True)]


def test_artifact_context_hook_finalizes_after_sglang_device_setup(
    armed_env, monkeypatch, fake_model_runner_module
):
    observed = fake_model_runner_module._calls
    monkeypatch.setattr(
        seam,
        "finalize_pending_candidate_bundle",
        lambda: observed.append("optima-finalize"),
    )
    artifact_context.install()
    assert artifact_context.is_installed()
    wrapped = fake_model_runner_module.ModelRunner.init_torch_distributed
    artifact_context.install()
    assert fake_model_runner_module.ModelRunner.init_torch_distributed is wrapped

    result = fake_model_runner_module.ModelRunner().init_torch_distributed()
    assert result == "memory-snapshot"
    assert observed == ["sglang-context", "optima-finalize"]

    observed.clear()
    fake_model_runner_module.ModelRunner(
        is_draft_worker=True
    ).init_torch_distributed()
    assert observed == ["sglang-context"]

    artifact_context.uninstall()
    assert not artifact_context.is_installed()


def test_artifact_context_is_inert_without_scheduler_pending_authority(
    armed_env, monkeypatch, fake_model_runner_module
):
    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("output-path process attempted candidate load")

    monkeypatch.setattr(seam, "_load_candidate_bundle_locked", forbidden_load)
    artifact_context.install()
    assert (
        fake_model_runner_module.ModelRunner().init_torch_distributed()
        == "memory-snapshot"
    )
    assert fake_model_runner_module._calls == ["sglang-context"]
    artifact_context.uninstall()
