from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

from optima.eval import engine_worker


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
    def test_process_hardening_requires_zero_caps_and_live_seccomp(self):
        with mock.patch.object(
            Path, "read_text", autospec=True, side_effect=_sandbox_proc_reader()
        ):
            self.assertTrue(engine_worker._process_sandbox_is_hardened())
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
                self.assertFalse(engine_worker._process_sandbox_is_hardened())


def test_engine_kwargs_preserve_candidate_overrides():
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
