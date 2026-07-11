import os
import sys
from types import SimpleNamespace
from unittest import TestCase, mock

import pytest

from optima import receipts, seam
from optima.eval import _launch


class IsolationTests(TestCase):
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
    monkeypatch.setattr(receipts, "require", lambda *_a, **_k: [{"slots": ["s"]}])

    with _launch.launched_engine(
        cfg, bundle_path="bundle" if active else "", active=active
    ):
        assert os.environ["OPTIMA_FRAMEWORK_MODE"] == expected

    assert observed == [expected]
    assert os.environ["OPTIMA_FRAMEWORK_MODE"] == "ambient"
