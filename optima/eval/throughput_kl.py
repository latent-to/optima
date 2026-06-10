"""End-to-end evaluation: throughput + output-distribution (KL) fidelity.

Two launches of the same model (identical weights/seed/sampler), differing only
by whether the miner kernel is enabled, isolate the kernel's effect: the
throughput delta is the kernel's, and the per-position KL between the two runs is
how much it perturbed the output. A faithful kernel yields KL ~ 0 and (hopefully)
speedup > 1.

Robustness measures (vs the first MVP):

* tamper-resistant timing — the driver process calls ``seam.mark_driver()`` so it
  never imports the miner module; the kernel runs only in the spawned scheduler,
  which the driver times over IPC. A malicious kernel cannot reach the clock.
* median-of-K — each launch times the workload K times and reports the median
  plus spread, so a single noisy sample can't swing the score.
* larger, seeded prompt set — sampled per epoch from a corpus so a kernel can't
  special-case a fixed handful of prompts, and more positions stabilize the KL.

GPU-only; imports sglang lazily.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from optima.eval._launch import call_in_subprocess
from optima.eval.kl import KLReport, aligned_kl, extract_per_prompt, kl_gate_ok, token_match_rate
from optima.eval.prompts import sample_prompts
from optima.eval.scoring import score_speedup


@dataclass
class EvalConfig:
    model_path: str
    dtype: str = "bfloat16"
    max_new_tokens: int = 64
    num_prompts: int = 32
    timed_iters: int = 3  # median-of-K timed passes per launch
    top_logprobs_num: int = 20
    temperature: float = 0.0  # greedy -> deterministic alignment
    # Fixed token budget by default: with greedy decode this forces baseline AND
    # candidate to emit EXACTLY max_new_tokens, so throughput is a pure per-token
    # latency comparison and a kernel can't inflate tok/s by nudging EOS timing
    # (the self-reported token count is no longer a lever). Turn off only for a
    # natural-length probe, never for scoring.
    ignore_eos: bool = True
    warmup_iters: int = 2  # >=2 full rounds: 1 leaves the documented ±17-32% clock-ramp in-window
    deterministic: bool = False
    # None -> advisory (KL reported but not gated; for big MoE where the
    # nondeterminism floor exceeds any sane threshold and accuracy carries quality).
    kl_threshold: Optional[float] = 5e-3
    # Sparse-cheat guards alongside mean_kl (active only when kl_threshold is set):
    #   argmax_disagree_rate catches a kernel that flips a few tokens while keeping
    #   the mean low; p99 catches a catastrophic tail. Calibrate to the noise floor —
    #   in deterministic mode a faithful kernel sits at 0 flips (see README).
    argmax_disagree_rate_threshold: Optional[float] = 0.01
    p99_kl_threshold: Optional[float] = None  # opt-in (needs per-model calibration)
    # Tail-mass guard: top-k KL is blind to mass moved into the unreported tail, so a
    # flattened/diversity-collapsed candidate with a matching head passes it. mean
    # coverage deviation catches that. Loose default (faithful kernels sit ~0); tighten
    # per model. None -> off.
    coverage_dev_threshold: Optional[float] = 0.25
    # FRAMEWORK MODE: when the miner may patch the engine (a setup() callable), its
    # self-reported logprobs are NOT trustworthy, so the quality gate switches from
    # in-process KL to TOKEN-MATCH vs the trusted stock baseline — the candidate's
    # emitted tokens are only correct if it actually computed correctly. Full
    # cheat-resistance also needs no-egress isolation (see the threat model docs).
    framework_mode: bool = False
    token_match_threshold: float = 0.99  # min fraction of generated tokens matching baseline
    # No-egress isolation for the CANDIDATE launch (the untrusted side): run it in a
    # fresh network namespace so miner code can't fetch the reference output. Required
    # for framework_mode to be cheat-PROOF (the cli turns it on with --framework-mode).
    isolate: bool = False
    # Dev-only escape hatch for pods that cannot create a netns. Production scoring
    # must leave this False so failed isolation is a hard error.
    allow_unsafe_no_isolation: bool = False
    seed: int = 0  # model seed
    prompt_seed: int = 0  # per-epoch prompt sampling seed
    # FLOOR on the required improvement (see optima/eval/scoring.py). The ACTUAL bar
    # is max(speedup_margin, score_k * measured_baseline_noise) — derived from the box,
    # not hand-picked — because a constant 2% sits an order of magnitude below the
    # ±7-17% warmup/thermal noise on a pod whose clocks we can't lock.
    speedup_margin: float = 0.02
    # Noise-robust scoring (we cannot lock GPU clocks on rented pods):
    #  * bookend_baseline: measure stock BEFORE and AFTER the candidate (B,C,B') so the
    #    candidate is bracketed; the two baseline reads bound the drift across it and
    #    give a per-round noise estimate. Off -> the old single-baseline 2-launch (cheap
    #    debug only; cannot be confident, so it never crowns).
    #  * score_k: how many measured-noise-widths above 1.0 a speedup must clear.
    #  * max_noise: if the bracketing baselines disagree by more than this, the round is
    #    untrustworthy -> NO-DECISION (never crowns), the subnet re-queues it.
    bookend_baseline: bool = True
    score_k: float = 2.0
    max_noise: float = 0.10
    # None -> sglang auto-picks the best backend for the hardware (fa3 on Hopper,
    # etc.). Don't hard-code a weak backend: a production-strong baseline is required,
    # or miners optimize against a slow reference. Override per-HW only if needed.
    attention_backend: Optional[str] = None
    # Graphs ON by default. Disabling CUDA graphs cripples the baseline (~6.5x slower
    # on 0.5B decode, measured on an H100), so a faithful kernel would "win" against a
    # weak reference. The seam is CUDA-graph-safe (validated). Set True only for quick
    # eager debugging, never for scoring.
    disable_cuda_graph: bool = False
    mem_fraction_static: float = 0.6
    log_level: str = "warning"
    # Serving regime: cap the concurrently-running requests so throughput is measured at a
    # production-like batch, not just whatever a single generate() call packs. The right
    # kernel is regime-dependent (low-batch=dispatch-bound, high-batch=memory-bound), so a
    # win must be measured at the serving operating point. None -> sglang default. PARTIAL
    # fix for the eval-vs-serving-distribution gap (report M2/#12): the knob exists; a full
    # per-epoch multi-regime sweep + worst-regime gate is still future work.
    max_running_requests: Optional[int] = None
    # multi-GPU knobs (TP size, MoE backend, custom-allreduce toggle for tensor-parallel
    # runs; see docs/DEV_ENVIRONMENT.md). Left unset by default so single-GPU runs are
    # byte-for-byte unchanged.
    tp_size: Optional[int] = None
    moe_runner_backend: Optional[str] = None
    disable_custom_all_reduce: bool = False
    candidate_attention_backend: Optional[str] = None
    candidate_moe_runner_backend: Optional[str] = None
    candidate_disable_custom_all_reduce: Optional[bool] = None
    extra_engine_kwargs: dict[str, Any] = field(default_factory=dict)
    candidate_extra_engine_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModeResult:
    tok_per_s: float  # median across timed_iters
    tok_per_s_samples: list[float]
    tokens: int
    per_prompt: list[tuple[list[int], list]]  # (output_ids, per-position top-k)

    @property
    def spread(self) -> tuple[float, float, float]:
        s = self.tok_per_s_samples
        if len(s) < 2:
            return (min(s, default=0.0), max(s, default=0.0), 0.0)
        return (min(s), max(s), statistics.pstdev(s))


@dataclass
class EvalReport:
    baseline: ModeResult
    candidate: ModeResult
    speedup: float  # informational: candidate / mean(bracketing baselines)
    kl: KLReport
    passed_quality: bool
    passed_speedup: bool  # NOISE-AWARE: cleared the measured bar AND the round was trustworthy
    score: float  # the crownable speedup (>=bar, confident) or 0.0 — what the ledger records
    token_match: float = 1.0  # fraction of tokens matching baseline (the framework-mode gate)
    noise: float = 0.0  # measured relative spread of the baseline reads
    required_speedup: float = 1.0  # the bar the speedup had to clear this round
    confident: bool = True  # False -> box too noisy this round; NO-DECISION, never crowns
    baseline2: Optional[ModeResult] = None  # the trailing bookend baseline (B'), if measured


@contextmanager
def _env(**overrides: str):
    import os

    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _sampling_params(cfg: EvalConfig) -> dict:
    sp = {"temperature": cfg.temperature, "max_new_tokens": cfg.max_new_tokens}
    if cfg.ignore_eos:
        sp["ignore_eos"] = True
    return sp


def _timed_generate(engine, prompts: list[str], cfg: EvalConfig, *, with_logprobs: bool):
    sp = _sampling_params(cfg)
    kwargs: dict[str, Any] = {}
    if with_logprobs:
        kwargs = dict(return_logprob=True, logprob_start_len=-1, top_logprobs_num=cfg.top_logprobs_num)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = engine.generate(prompt=list(prompts), sampling_params=sp, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if isinstance(outputs, dict):
        outputs = [outputs]
    tokens = _counted_tokens(outputs, prompts, cfg)
    return outputs, tokens, elapsed


def _counted_tokens(outputs, prompts, cfg) -> int:
    """The throughput numerator. The token COUNT is produced in the scheduler process
    where the miner kernel also runs, so it isn't trustworthy on its own. Under the
    scoring default (ignore_eos + a fixed max_new_tokens) the driver knows the count a
    PRIORI — ``len(prompts) * max_new_tokens`` — so we use that and never trust a
    scheduler-reported field. Only when natural-length generation is explicitly
    requested (--no-ignore-eos) do we fall back to the reported completion_tokens."""
    if getattr(cfg, "ignore_eos", False):
        return len(prompts) * int(cfg.max_new_tokens)
    return sum(int(o.get("meta_info", {}).get("completion_tokens", 0)) for o in outputs)


def _measure(engine, prompts: list[str], cfg: EvalConfig) -> ModeResult:
    # Warmup (JIT/compile/graph) off the clock.
    for _ in range(max(0, cfg.warmup_iters)):
        engine.generate(prompt=list(prompts), sampling_params=_sampling_params(cfg))

    samples: list[float] = []
    last_outputs = None
    last_tokens = 0
    for i in range(max(1, cfg.timed_iters)):
        # Capture logprobs only on the last iter (cheaper, and the dist is stable).
        with_lp = i == cfg.timed_iters - 1
        outputs, tokens, elapsed = _timed_generate(engine, prompts, cfg, with_logprobs=with_lp)
        if elapsed > 0:
            samples.append(tokens / elapsed)
        if with_lp:
            last_outputs, last_tokens = outputs, tokens

    return ModeResult(
        tok_per_s=statistics.median(samples) if samples else 0.0,
        tok_per_s_samples=samples,
        tokens=last_tokens,
        per_prompt=extract_per_prompt(last_outputs or []),
    )


def _run_launch(cfg: EvalConfig, prompts: list[str], *, bundle_path: str, active: bool) -> ModeResult:
    # Mark THIS process as the timer/driver before importing sglang, so the seam
    # here is pass-through only and the miner module is never imported in the
    # process that measures wall-clock.
    from optima import seam

    seam.mark_driver()

    from optima.eval._launch import engine_kwargs, prepare_candidate_environment

    prepare_candidate_environment(cfg, bundle_path=bundle_path, active=active)

    with _env(
        OPTIMA_BUNDLE_PATH=bundle_path or "",
        OPTIMA_ACTIVE="1" if active else "0",
        SGLANG_PLUGINS="optima",
    ):
        import sglang as sgl

        engine = sgl.Engine(**engine_kwargs(cfg, active=active))
        try:
            return _measure(engine, prompts, cfg)
        finally:
            try:
                engine.shutdown()
            except Exception:  # noqa: BLE001
                pass


def _aligned_kl(baseline: ModeResult, candidate: ModeResult) -> KLReport:
    # Per-prompt alignment up to the first token divergence; see kl.aligned_kl.
    return aligned_kl(baseline.per_prompt, candidate.per_prompt)


def evaluate(cfg: EvalConfig, bundle_path: str, prompts: Optional[list[str]] = None) -> EvalReport:
    prompts = list(prompts) if prompts else sample_prompts(cfg.num_prompts, cfg.prompt_seed)

    # Bookended A/B (we cannot lock GPU clocks on rented pods): measure stock BEFORE
    # and AFTER the candidate so the candidate is bracketed and the two baseline reads
    # bound the warmup/thermal drift across it. Each launch runs in its own fresh
    # process (call_in_subprocess) so the baseline's deterministic/CUDA global state
    # can't corrupt the candidate. See optima/eval/scoring.py.
    baseline = call_in_subprocess(_run_launch, cfg, prompts, bundle_path="", active=False)
    candidate = call_in_subprocess(_run_launch, cfg, prompts, bundle_path=bundle_path, active=True)
    baseline2 = (call_in_subprocess(_run_launch, cfg, prompts, bundle_path="", active=False)
                 if cfg.bookend_baseline else None)

    baseline_reads = [baseline.tok_per_s] + ([baseline2.tok_per_s] if baseline2 else [])
    verdict = score_speedup(
        baseline_reads, candidate.tok_per_s,
        min_margin=cfg.speedup_margin, k=cfg.score_k, max_noise=cfg.max_noise,
    )

    # KL/token fidelity vs the (stock) baseline — any stock run is a valid reference;
    # use the first so it's deterministic.
    kl = _aligned_kl(baseline, candidate)
    matched, total = token_match_rate(baseline.per_prompt, candidate.per_prompt)
    token_match = (matched / total) if total else 1.0
    if getattr(cfg, "framework_mode", False):
        # The miner may have patched the engine (setup()), so its self-reported logprobs
        # are not trusted: gate on token-match vs the trusted stock baseline, not KL.
        passed_quality = total > 0 and token_match >= cfg.token_match_threshold
    else:
        passed_quality = kl.num_positions > 0 and kl_gate_ok(
            kl,
            kl_threshold=cfg.kl_threshold,
            p99_kl_threshold=cfg.p99_kl_threshold,
            argmax_disagree_rate_threshold=cfg.argmax_disagree_rate_threshold,
            coverage_dev_threshold=cfg.coverage_dev_threshold,
        )
    # Crownable only when quality holds AND the speedup is a noise-confident real win.
    # The ledger records the speedup only when crownable, else 0.0 — so a cheat (quality
    # fail), a faithful-but-not-faster kernel, OR a too-noisy round can never take the
    # title. The raw speedup is still reported for the human read.
    crownable = passed_quality and verdict.passed_speedup
    score = verdict.speedup if crownable else 0.0

    return EvalReport(
        baseline, candidate, verdict.speedup, kl, passed_quality, verdict.passed_speedup, score,
        token_match, noise=verdict.noise, required_speedup=verdict.required,
        confident=verdict.confident, baseline2=baseline2,
    )
