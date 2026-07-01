"""Benchmark-based capability eval — the score is throughput, gated by task
performance on real benchmarks.

This implements the design: the quality gate is not (only) KL — it's "did the
model's accuracy on real benchmark problems survive the kernel?" We run the same
fixed benchmark sample through two launches (kernel off = baseline, kernel on =
candidate), measure throughput and per-benchmark accuracy for each, and gate:

* **quality**: two gates on the SAME realistic run — (1) no accuracy regression on
  ANY benchmark beyond a small tolerance (Affine's "strictly not worse across all
  envs"), and (2) per-token KL vs the baseline under threshold (the dense,
  low-variance check; accuracy at small n is noisy, so KL is the primary gate).
* **score**: the single thing maximized is THROUGHPUT speedup. The benchmarks are
  pass/fail GATES, not score components — so there's nothing to aggregate with a
  geometric mean. Our objective is scalar (speed); correctness is a constraint.

A faithful kernel preserves accuracy and (hopefully) speeds things up. A kernel
that secretly degrades the model drops benchmark accuracy and scores zero, even
if it looked fast.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import torch

from optima.eval._launch import call_in_subprocess, launched_engine
from optima.eval.benchmarks import Problem, get_benchmark
from optima.eval.kl import KLReport, aligned_kl, extract_per_prompt, kl_gate_ok, token_match_rate
from optima.eval.scoring import score_speedup
from optima.eval.throughput_kl import EvalConfig


@dataclass
class BenchmarkScore:
    name: str
    n: int
    baseline_correct: int
    candidate_correct: int

    @property
    def baseline_acc(self) -> float:
        return self.baseline_correct / self.n if self.n else 0.0

    @property
    def candidate_acc(self) -> float:
        return self.candidate_correct / self.n if self.n else 0.0

    @property
    def delta(self) -> float:
        return self.candidate_acc - self.baseline_acc


@dataclass
class CapabilityReport:
    benchmarks: list[BenchmarkScore]
    baseline_tok_s: float
    candidate_tok_s: float
    speedup: float
    passed_quality: bool
    passed_speedup: bool  # NOISE-AWARE (see optima/eval/scoring.py)
    score: float  # crownable speedup or 0.0 — what the ledger records
    kl: KLReport
    token_match: float = 1.0
    regressions: list[str] = field(default_factory=list)
    noise: float = 0.0
    required_speedup: float = 1.0
    confident: bool = True
    baseline2_tok_s: float = 0.0


def _sampling_params(cfg: EvalConfig, *, max_new_tokens: int) -> dict:
    sp = {"temperature": 0.0, "max_new_tokens": max_new_tokens}
    if getattr(cfg, "ignore_eos", False):
        sp["ignore_eos"] = True
    return sp


def _generate_and_time(engine, prompts: list[str], *, max_new_tokens: int, timed_iters: int,
                       cfg: EvalConfig, top_logprobs_num: int = 0):
    sp = _sampling_params(cfg, max_new_tokens=max_new_tokens)
    # warmup (JIT/compile AND the clock ramp off the clock): cfg.warmup_iters full
    # rounds, >=2 — a single round leaves the documented ±17-32% ramp inside the
    # timing window (see EvalConfig.warmup_iters).
    for _ in range(max(1, getattr(cfg, "warmup_iters", 1))):
        engine.generate(prompt=prompts, sampling_params=sp)

    samples: list[float] = []
    outputs = None
    for i in range(max(1, timed_iters)):
        # Capture top-k logprobs on the last timed iter only (cheaper) so KL is
        # computed on the SAME realistic run we time and answer-check.
        with_lp = top_logprobs_num > 0 and i == timed_iters - 1
        kwargs = (dict(return_logprob=True, logprob_start_len=-1, top_logprobs_num=top_logprobs_num)
                  if with_lp else {})
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = engine.generate(prompt=prompts, sampling_params=sp, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if isinstance(outs, dict):
            outs = [outs]
        # Numerator: under ignore_eos the driver knows the count a priori (fixed budget),
        # so don't trust the scheduler-reported completion_tokens (#11).
        if getattr(cfg, "ignore_eos", False):
            tokens = len(prompts) * int(max_new_tokens)
        else:
            tokens = sum(int(o.get("meta_info", {}).get("completion_tokens", 0)) for o in outs)
        if elapsed > 0:
            samples.append(tokens / elapsed)
        outputs = outs  # keep last for answer-checking + KL
    tok_s = statistics.median(samples) if samples else 0.0
    texts = [o.get("text", "") for o in (outputs or [])]
    per_prompt = extract_per_prompt(outputs or [])
    return tok_s, texts, per_prompt


def _run_launch(cfg: EvalConfig, flat: list[tuple[str, Problem]], *, bundle_path: str, active: bool,
                max_new_tokens: int):
    prompts = [p.prompt for _, p in flat]
    with launched_engine(cfg, bundle_path=bundle_path, active=active) as engine:
        return _generate_and_time(engine, prompts, max_new_tokens=max_new_tokens,
                                  timed_iters=cfg.timed_iters,
                                  top_logprobs_num=cfg.top_logprobs_num,
                                  cfg=cfg)


def _accuracy_by_benchmark(flat: list[tuple[str, Problem]], texts: list[str]) -> dict[str, int]:
    correct: dict[str, int] = {}
    for (bench_name, problem), text in zip(flat, texts):
        bench = get_benchmark(bench_name)
        if bench.check(problem, text):
            correct[bench_name] = correct.get(bench_name, 0) + 1
        else:
            correct.setdefault(bench_name, 0)
    return correct


def evaluate_capability(
    cfg: EvalConfig,
    bundle_path: str,
    benchmark_names: list[str],
    *,
    samples_per_benchmark: int = 32,
    acc_tolerance: float = 0.02,
    max_new_tokens: int | None = None,
) -> CapabilityReport:
    # Build one flat, ordered list of (benchmark, problem); max_new_tokens is the
    # max any benchmark needs (they're generated together).
    flat: list[tuple[str, Problem]] = []
    counts: dict[str, int] = {}
    max_new = 0
    for name in benchmark_names:
        bench = get_benchmark(name)
        probs = bench.load(samples_per_benchmark, cfg.prompt_seed)
        counts[name] = len(probs)
        max_new = max(max_new, bench.max_new_tokens)
        flat.extend((name, p) for p in probs)
    if max_new_tokens is not None:
        max_new = int(max_new_tokens)

    # Bookended A/B (clocks unlockable on rented pods): stock BEFORE and AFTER the
    # candidate brackets it and bounds the drift across it. Each launch runs in its
    # own fresh process so the baseline's deterministic/CUDA global state can't corrupt
    # the candidate (see _launch.call_in_subprocess + optima/eval/scoring.py).
    base_tok_s, base_texts, base_pp = call_in_subprocess(
        _run_launch, cfg, flat, bundle_path="", active=False, max_new_tokens=max_new)
    cand_tok_s, cand_texts, cand_pp = call_in_subprocess(
        _run_launch, cfg, flat, bundle_path=bundle_path, active=True, max_new_tokens=max_new)
    base2_tok_s = 0.0
    if getattr(cfg, "bookend_baseline", True):
        base2_tok_s, _b2_texts, _b2_pp = call_in_subprocess(
            _run_launch, cfg, flat, bundle_path="", active=False, max_new_tokens=max_new)

    base_correct = _accuracy_by_benchmark(flat, base_texts)
    cand_correct = _accuracy_by_benchmark(flat, cand_texts)

    scores: list[BenchmarkScore] = []
    regressions: list[str] = []
    for name in benchmark_names:
        bs = BenchmarkScore(
            name=name, n=counts[name],
            baseline_correct=base_correct.get(name, 0),
            candidate_correct=cand_correct.get(name, 0),
        )
        scores.append(bs)
        if bs.delta < -acc_tolerance:  # regressed beyond tolerance
            regressions.append(f"{name}: {bs.baseline_acc:.1%} -> {bs.candidate_acc:.1%}")

    # Dense fidelity gate on the SAME realistic prompts: KL between the two runs.
    # mean_kl (diffuse drift) AND p99/argmax-rate (sparse cheats) — see kl_gate_ok.
    #  * kl_threshold is None  -> advisory: report KL but don't gate. Use on big MoE
    #    where the nondeterminism floor exceeds any sane threshold; accuracy carries it.
    #  * num_positions == 0    -> no logprobs; fall back to the accuracy floor.
    kl = aligned_kl(base_pp, cand_pp)
    matched, total = token_match_rate(base_pp, cand_pp)
    token_match = (matched / total) if total else 1.0
    kl_ok = kl_gate_ok(
        kl,
        kl_threshold=cfg.kl_threshold,
        p99_kl_threshold=cfg.p99_kl_threshold,
        argmax_disagree_rate_threshold=cfg.argmax_disagree_rate_threshold,
        coverage_dev_threshold=getattr(cfg, "coverage_dev_threshold", None),
    )

    baseline_reads = [base_tok_s] + ([base2_tok_s] if base2_tok_s > 0 else [])
    verdict = score_speedup(
        baseline_reads, cand_tok_s,
        min_margin=cfg.speedup_margin, k=getattr(cfg, "score_k", 2.0),
        max_noise=getattr(cfg, "max_noise", 0.10),
    )
    if getattr(cfg, "framework_mode", False):
        fidelity_ok = total > 0 and token_match >= cfg.token_match_threshold
    else:
        fidelity_ok = kl_ok
    passed_quality = len(regressions) == 0 and fidelity_ok
    crownable = passed_quality and verdict.passed_speedup
    score = verdict.speedup if crownable else 0.0

    return CapabilityReport(
        benchmarks=scores,
        baseline_tok_s=base_tok_s,
        candidate_tok_s=cand_tok_s,
        speedup=verdict.speedup,
        passed_quality=passed_quality,
        passed_speedup=verdict.passed_speedup,
        score=score,
        kl=kl,
        token_match=token_match,
        regressions=regressions,
        noise=verdict.noise,
        required_speedup=verdict.required,
        confident=verdict.confident,
        baseline2_tok_s=base2_tok_s,
    )
