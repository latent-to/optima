from __future__ import annotations

from types import SimpleNamespace

from optima.eval import throughput_kl
from optima.eval.throughput_kl import EvalConfig, ModeResult, effective_fidelity_mode


def _mode(token: int, rate: float) -> ModeResult:
    topk = [[(0.0, token, None)]]
    return ModeResult(
        tok_per_s=rate,
        tok_per_s_samples=[rate, rate],
        tokens=1,
        per_prompt=[([token], topk)],
    )


def test_framework_fidelity_overrides_requested_audit():
    assert effective_fidelity_mode(
        SimpleNamespace(framework_mode=True, fidelity_mode="audit")
    ) == "framework"
    assert effective_fidelity_mode(
        SimpleNamespace(framework_mode=False, fidelity_mode="audit")
    ) == "audit"


def test_framework_mode_skips_in_engine_audit_and_uses_external_tokens(monkeypatch):
    baseline = _mode(7, 100.0)
    candidate = _mode(8, 110.0)
    launches = []
    results = iter((baseline, candidate, baseline))

    def fake_call(fn, *_args, **_kwargs):
        launches.append(fn.__name__)
        return next(results)

    monkeypatch.setattr(throughput_kl, "call_in_subprocess", fake_call)
    report = throughput_kl.evaluate(
        EvalConfig(
            model_path="model",
            num_prompts=1,
            max_new_tokens=1,
            timed_iters=1,
            warmup_iters=0,
            framework_mode=True,
            fidelity_mode="audit",
            token_match_threshold=1.0,
        ),
        "bundle",
        prompts=["prompt"],
    )

    assert launches == ["_run_launch", "_run_launch", "_run_launch"]
    assert report.fidelity_mode == "framework"
    assert report.token_match == 0.0
    assert not report.passed_quality
    assert report.score == 0.0
