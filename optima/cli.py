"""Optima validator CLI — drives the submission pipeline end to end.

    python -m optima.cli slots
    python -m optima.cli scan      <bundle>
    python -m optima.cli verify    <bundle> [--dtype bfloat16] [--device cuda]
    python -m optima.cli evaluate  <bundle> --model <path> [--max-new-tokens 128]

Pipeline (mirrors the validator flow):

    manifest -> static scan -> (isolated) load -> op-correctness -> register
             -> build engine -> baseline vs candidate -> throughput + KL -> score

SECURITY NOTE: ``verify`` and ``evaluate`` import the miner module, which runs
its code in THIS process. That is only acceptable because the whole validator
host is expected to be the sandbox (no network, per-eval GPU context, watchdog).
Do not run this on a machine you care about without that isolation. See
``optima/sandbox.py``.
"""

from __future__ import annotations

import argparse
import json
import sys

from optima.manifest import load_manifest, resolve_source
from optima.sandbox import load_entry, scan_path


def _json_obj(raw: str | None) -> dict:
    if not raw:
        return {}
    out = json.loads(raw)
    if not isinstance(out, dict):
        raise argparse.ArgumentTypeError("JSON value must be an object")
    return out


def _dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def cmd_slots(_: argparse.Namespace) -> int:
    from optima.slots import SLOTS, list_slots

    print("Registered op-slots (the submission ABI):")
    for name in list_slots():
        spec = SLOTS[name]
        print(f"  {name}  [{spec.kind}]")
        print(f"      {spec.summary}")
    return 0


def cmd_compat(_: argparse.Namespace) -> int:
    from optima.compat import format_checks, run_checks

    checks = run_checks()
    print("sglang compatibility canary (run after any sglang bump):")
    print(format_checks(checks))
    return 0 if all(c.ok for c in checks) else 2


def cmd_chain_compat(_: argparse.Namespace) -> int:
    from optima.chain_canary import format_checks, run_checks

    checks = run_checks()
    print("bittensor chain-SDK canary (introspects the installed SDK; no network):")
    print(format_checks(checks))
    return 0 if all(c.ok for c in checks) else 2


def cmd_set_weights(args: argparse.Namespace) -> int:
    from optima import chain
    from optima.commit_reveal import Ledger

    led = Ledger.load(args.ledger)
    if not led.champion:
        print(f"no champion in {args.ledger}; nothing to weight")
        return 1
    weights = {led.champion.hotkey: 1.0}  # winner-take-all baseline (matches settle)
    subtensor = chain.connect(args.network)
    if args.dry_run:
        res = chain.set_weights(subtensor, None, args.netuid, weights, dry_run=True)
        print(f"DRY RUN (network={args.network}, netuid={args.netuid}): "
              f"would set uids={res['uids']} weights={res['weights']}")
        return 0
    import bittensor as bt

    wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
    res = chain.set_weights(subtensor, wallet, args.netuid, weights)
    print(f"set_weights submitted={res.get('submitted')} uids={res.get('uids')}")
    return 0 if res.get("submitted") else 1


def cmd_scan(args: argparse.Namespace) -> int:
    m = load_manifest(args.bundle)
    print(f"bundle: {m.bundle_id}  abi: {m.abi_version}  ops: {len(m.ops)}")
    rc = 0
    for op in m.ops:
        src = resolve_source(args.bundle, op)
        result = scan_path(src)
        status = "clean" if result.ok else "VIOLATIONS"
        print(f"  [{status}] {op.slot} <- {op.source}")
        for v in result.violations:
            print(f"      {v}")
            rc = 2
    return rc


def cmd_verify(args: argparse.Namespace) -> int:
    from optima.slots import SLOTS, get_slot
    from optima.verify import format_verify, verify_entry

    m = load_manifest(args.bundle)
    rc = 0
    for op in m.ops:
        if op.slot not in SLOTS:
            print(f"  [SKIP] {op.slot}: not a known slot on this validator")
            continue
        slot = get_slot(op.slot)
        src = resolve_source(args.bundle, op)

        scan = scan_path(src)
        if not scan.ok:
            print(f"  [FAIL] {op.slot}: failed policy scan")
            for v in scan.violations:
                print(f"      {v}")
            rc = 2
            continue

        if slot.kind == "collective":
            # Collectives span ranks -> distributed verify (spawns world_size ranks;
            # gloo/CPU if device=cpu, nccl/GPU if cuda). No per-op single-process path.
            from optima.verify_collective import verify_collective

            ws = getattr(args, "world_size", None) or 2
            result = verify_collective(slot, str(src), op.entry, world_size=ws, device=args.device, seed=args.seed)
            print(format_verify(result))
            if not result.passed:
                rc = 2
            continue

        entry = load_entry(src, op.entry)  # SECURITY: isolate in production
        prepare_fn = load_entry(src, op.prepare) if op.prepare else None  # (prepare, forward) slots
        result = verify_entry(
            slot, entry, prepare=prepare_fn, dtype=_dtype(args.dtype), device=args.device, seed=args.seed
        )
        print(format_verify(result))
        if not result.passed:
            rc = 2
    return rc


def cmd_evaluate(args: argparse.Namespace) -> int:
    from optima.slots import SLOTS
    from optima.eval.throughput_kl import EvalConfig, evaluate

    # Trusted parent: validate + scan only. It never imports miner code — the
    # kernel is loaded inside the (to-be-isolated) model process by the plugin.
    m = load_manifest(args.bundle)
    known = 0
    for op in m.ops:
        if op.slot not in SLOTS:
            print(f"  [skip] {op.slot}: unknown slot")
            continue
        src = resolve_source(args.bundle, op)
        scan = scan_path(src)
        if not scan.ok:
            print(f"  [FAIL] {op.slot}: failed policy scan; aborting")
            for v in scan.violations:
                print(f"      {v}")
            return 2
        known += 1
        print(f"  [ok]   {op.slot} <- {op.source} ({op.entry}) [scan clean]")

    if known == 0:
        print("no known slots in this bundle; nothing to evaluate")
        return 1

    cfg = EvalConfig(
        model_path=args.model,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        num_prompts=args.num_prompts,
        framework_mode=args.framework_mode,
        token_match_threshold=args.token_match_threshold,
        isolate=args.isolate or args.framework_mode,  # framework-mode implies no-egress isolation
        allow_unsafe_no_isolation=args.allow_unsafe_no_isolation,
        timed_iters=args.timed_iters,
        prompt_seed=args.prompt_seed,
        top_logprobs_num=args.top_logprobs,
        ignore_eos=args.ignore_eos,
        kl_threshold=None if args.kl_advisory else args.kl_threshold,
        argmax_disagree_rate_threshold=args.argmax_disagree_rate,
        p99_kl_threshold=args.p99_kl_threshold,
        deterministic=not args.no_deterministic,
        attention_backend=args.attention_backend,
        disable_cuda_graph=args.disable_cuda_graph,
        mem_fraction_static=args.mem_fraction,
        tp_size=args.tp_size,
        moe_runner_backend=args.moe_runner_backend,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        candidate_attention_backend=args.candidate_attention_backend,
        candidate_moe_runner_backend=args.candidate_moe_runner_backend,
        candidate_disable_custom_all_reduce=args.candidate_disable_custom_all_reduce,
        extra_engine_kwargs=_json_obj(args.engine_kwargs_json),
        candidate_extra_engine_kwargs=_json_obj(args.candidate_engine_kwargs_json),
    )
    print(f"\nrunning two launches of {args.model} (dtype={args.dtype}, "
          f"deterministic={cfg.deterministic}, cuda_graph={not cfg.disable_cuda_graph}, "
          f"attn_backend={cfg.attention_backend or 'auto'}, "
          f"framework_mode={cfg.framework_mode}, isolate_candidate={cfg.isolate}, "
          f"unsafe_no_isolation={cfg.allow_unsafe_no_isolation}): "
          f"baseline then candidate ...")
    report = evaluate(cfg, str(args.bundle))

    b, c = report.baseline, report.candidate
    bmin, bmax, bsd = b.spread
    cmin, cmax, csd = c.spread
    print("\n=== Optima end-to-end report ===")
    print(f"bundle: {m.bundle_id}")
    print(f"baseline   {b.tok_per_s:8.1f} tok/s  (median of {len(b.tok_per_s_samples)}; "
          f"range {bmin:.0f}-{bmax:.0f}, sd {bsd:.1f})")
    print(f"candidate  {c.tok_per_s:8.1f} tok/s  (median of {len(c.tok_per_s_samples)}; "
          f"range {cmin:.0f}-{cmax:.0f}, sd {csd:.1f})")
    print(f"speedup    {report.speedup:8.3f}x  (needs >= {1 + cfg.speedup_margin:.2f} -> "
          f"{'PASS' if report.passed_speedup else 'below margin'})")
    print(f"quality    mean_kl={report.kl.mean_kl:.3e} max_kl={report.kl.max_kl:.3e} "
          f"argmax_disagree={report.kl.argmax_disagreements}/{report.kl.num_positions}  "
          f"token_match={report.token_match:.4f}{' (GATE)' if cfg.framework_mode else ''} -> "
          f"{'PASS' if report.passed_quality else 'FAIL'}")
    print(f"SCORE      {report.score:.3f}")

    if getattr(args, "ledger", None) and getattr(args, "hotkey", None):
        from optima.bundle_hash import content_hash
        from optima.commit_reveal import Ledger

        ch = content_hash(args.bundle)
        led = Ledger.load(args.ledger)
        led.record_score(args.hotkey, ch, args.round, report.score, report.kl.mean_kl, report.passed_quality)
        led.save(args.ledger)
        print(f"recorded -> {args.ledger} (hotkey={args.hotkey}, round={args.round})")
    return 0 if report.passed_quality else 3


def cmd_bench(args: argparse.Namespace) -> int:
    from optima.slots import SLOTS
    from optima.eval.capability import evaluate_capability
    from optima.eval.throughput_kl import EvalConfig

    m = load_manifest(args.bundle)
    known = 0
    for op in m.ops:
        if op.slot not in SLOTS:
            continue
        src = resolve_source(args.bundle, op)
        scan = scan_path(src)
        if not scan.ok:
            print(f"  [FAIL] {op.slot}: failed policy scan; aborting")
            for v in scan.violations:
                print(f"      {v}")
            return 2
        known += 1
    if known == 0:
        print("no known slots in this bundle; nothing to evaluate")
        return 1

    cfg = EvalConfig(
        model_path=args.model,
        dtype=args.dtype,
        timed_iters=args.timed_iters,
        prompt_seed=args.prompt_seed,
        top_logprobs_num=args.top_logprobs,
        ignore_eos=args.ignore_eos,
        kl_threshold=None if args.kl_advisory else args.kl_threshold,
        argmax_disagree_rate_threshold=args.argmax_disagree_rate,
        p99_kl_threshold=args.p99_kl_threshold,
        framework_mode=args.framework_mode,
        token_match_threshold=args.token_match_threshold,
        isolate=args.isolate or args.framework_mode,
        allow_unsafe_no_isolation=args.allow_unsafe_no_isolation,
        deterministic=not args.no_deterministic,
        attention_backend=args.attention_backend,
        disable_cuda_graph=args.disable_cuda_graph,
        mem_fraction_static=args.mem_fraction,
        tp_size=args.tp_size,
        moe_runner_backend=args.moe_runner_backend,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        candidate_attention_backend=args.candidate_attention_backend,
        candidate_moe_runner_backend=args.candidate_moe_runner_backend,
        candidate_disable_custom_all_reduce=args.candidate_disable_custom_all_reduce,
        extra_engine_kwargs=_json_obj(args.engine_kwargs_json),
        candidate_extra_engine_kwargs=_json_obj(args.candidate_engine_kwargs_json),
    )
    names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    print(f"\nbenchmark eval of {args.model} on {names} "
          f"({args.samples}/bench; framework_mode={cfg.framework_mode}, "
          f"isolate_candidate={cfg.isolate}, "
          f"unsafe_no_isolation={cfg.allow_unsafe_no_isolation}): baseline then candidate ...")
    report = evaluate_capability(
        cfg, str(args.bundle), names,
        samples_per_benchmark=args.samples, acc_tolerance=args.acc_tolerance,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n=== Optima capability report ===")
    print(f"bundle: {m.bundle_id}")
    for bs in report.benchmarks:
        flag = "" if bs.delta >= -args.acc_tolerance else "  <-- REGRESSION"
        print(f"  {bs.name:10s} baseline {bs.baseline_acc:6.1%} ({bs.baseline_correct}/{bs.n})  "
              f"candidate {bs.candidate_acc:6.1%} ({bs.candidate_correct}/{bs.n})  "
              f"Δ{bs.delta:+.1%}{flag}")
    print(f"throughput baseline {report.baseline_tok_s:8.1f} tok/s  candidate {report.candidate_tok_s:8.1f} tok/s")
    print(f"speedup    {report.speedup:8.3f}x  -> {'PASS' if report.passed_speedup else 'below margin'}")
    kl = report.kl
    if args.kl_advisory:
        kl_note = "advisory (not gated)"
    elif kl.num_positions == 0:
        kl_note = "n/a (no logprobs)"
    else:
        kl_note = f"<= {args.kl_threshold:.1e}"
    rate_note = "" if args.kl_advisory else f" (<= {args.argmax_disagree_rate:.1%})"
    print(f"quality    no-accuracy-regression + KL mean_kl={kl.mean_kl:.3e} ({kl_note}), "
          f"argmax_disagree={kl.argmax_disagreements}/{kl.num_positions} "
          f"({kl.argmax_disagree_rate:.2%}{rate_note}), "
          f"token_match={report.token_match:.4f}{' (GATE)' if cfg.framework_mode else ''} -> "
          f"{'PASS' if report.passed_quality else 'FAIL'}")
    print(f"SCORE      {report.score:.3f}")

    if getattr(args, "ledger", None) and getattr(args, "hotkey", None):
        from optima.bundle_hash import content_hash
        from optima.commit_reveal import Ledger

        ch = content_hash(args.bundle)
        led = Ledger.load(args.ledger)
        led.record_score(args.hotkey, ch, args.round, report.score, report.kl.mean_kl, report.passed_quality)
        led.save(args.ledger)
        print(f"recorded -> {args.ledger} (hotkey={args.hotkey}, round={args.round})")
    return 0 if report.passed_quality else 3


def cmd_hash(args: argparse.Namespace) -> int:
    from optima.bundle_hash import content_hash

    print(content_hash(args.bundle))
    return 0


def cmd_commit(args: argparse.Namespace) -> int:
    from optima.bundle_hash import content_hash
    from optima.commit_reveal import Ledger, make_commitment

    ch = content_hash(args.bundle)
    com = make_commitment(ch, args.hotkey, args.salt)
    led = Ledger.load(args.ledger)
    seq = led.commit(args.hotkey, com, args.round)
    led.save(args.ledger)
    print(f"committed hotkey={args.hotkey} round={args.round} seq={seq}")
    print(f"commitment={com}")
    print("keep your --salt and bundle; you'll need both to reveal")
    return 0


def cmd_reveal(args: argparse.Namespace) -> int:
    from optima.bundle_hash import content_hash
    from optima.commit_reveal import Ledger, RevealError

    ch = content_hash(args.bundle)
    led = Ledger.load(args.ledger)
    try:
        rev = led.reveal(args.hotkey, ch, args.salt, args.round)
    except RevealError as e:
        print(f"REJECTED: {e}")
        return 2
    led.save(args.ledger)
    print(f"revealed hotkey={args.hotkey} content={ch[:16]}... original={rev.original}")
    if not rev.original:
        print("  -> flagged as a COPY (an earlier commitment to this content exists); earns 0")
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    from optima.commit_reveal import Ledger

    led = Ledger.load(args.ledger)
    print(f"commitments={len(led.commitments)} reveals={len(led.reveals)} scores={len(led.scores)}")
    if led.champion:
        c = led.champion
        print(f"champion: hotkey={c.hotkey} score={c.score:.3f} round={c.round_id} "
              f"content={c.content_hash[:16]}...")
    else:
        print("champion: (none yet)")
    return 0


def cmd_settle(args: argparse.Namespace) -> int:
    from optima.commit_reveal import Ledger

    led = Ledger.load(args.ledger)
    res = led.settle(args.round, margin=args.margin)
    led.save(args.ledger)
    print(f"title_changed={res.title_changed} challenger_score={res.challenger_score:.3f}")
    if res.champion:
        print(f"champion: {res.champion.hotkey} score={res.champion.score:.3f}")
    if res.rejected_copies:
        print(f"rejected copies: {', '.join(res.rejected_copies)}")
    print(f"weights: {res.weights}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="optima", description="Optima validator harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("slots", help="list the op-slot ABI")
    sp.set_defaults(func=cmd_slots)

    sp = sub.add_parser("compat", help="check our sglang integration points survived an upgrade")
    sp.set_defaults(func=cmd_compat)

    sp = sub.add_parser("chain-compat",
                        help="check the installed bittensor SDK exposes the chain API we use")
    sp.set_defaults(func=cmd_chain_compat)

    sp = sub.add_parser("set-weights",
                        help="push the ledger champion's weights on-chain (king of the hill)")
    sp.add_argument("--ledger", default="optima_ledger.json")
    sp.add_argument("--netuid", type=int, required=True)
    sp.add_argument("--network", default="finney", help="'test' for the public testnet")
    sp.add_argument("--wallet", default="default")
    sp.add_argument("--hotkey", default="default")
    sp.add_argument("--dry-run", action="store_true",
                    help="build + print the (uids, weights) payload, do NOT submit")
    sp.set_defaults(func=cmd_set_weights)

    sp = sub.add_parser("scan", help="static policy scan of a bundle")
    sp.add_argument("bundle")
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("verify", help="op-level correctness vs reference")
    sp.add_argument("bundle")
    sp.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    sp.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("evaluate", help="end-to-end throughput + KL on a model")
    sp.add_argument("bundle")
    sp.add_argument("--model", required=True, help="model path for sglang.Engine")
    sp.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    sp.add_argument("--max-new-tokens", type=int, default=64)
    sp.add_argument("--num-prompts", type=int, default=32)
    sp.add_argument("--timed-iters", type=int, default=3, help="median-of-K timed passes per launch")
    sp.add_argument("--prompt-seed", type=int, default=0, help="per-epoch prompt sampling seed")
    sp.add_argument("--top-logprobs", type=int, default=20)
    sp.add_argument("--ignore-eos", action="store_true",
                    help="force generation to the max token budget; useful for long-decode throughput probes")
    sp.add_argument("--kl-threshold", type=float, default=5e-3)
    sp.add_argument("--argmax-disagree-rate", type=float, default=0.01,
                    help="max fraction of positions whose top token may flip (sparse-cheat guard)")
    sp.add_argument("--p99-kl-threshold", type=float, default=None, help="optional p99 KL gate (catastrophic tail)")
    sp.add_argument("--kl-advisory", action="store_true", help="report KL but don't gate on it")
    sp.add_argument("--mem-fraction", type=float, default=0.6,
                    help="sglang mem_fraction_static (use ~0.9 for big models like gpt-oss-120b)")
    sp.add_argument("--no-deterministic", action="store_true")
    sp.add_argument("--attention-backend", default=None,
                    help="sglang attention backend (default: auto-pick best per-HW, e.g. fa3/flashinfer)")
    sp.add_argument("--candidate-attention-backend", default=None,
                    help="candidate-only attention backend override")
    sp.add_argument("--disable-cuda-graph", action="store_true",
                    help="eager mode for quick debugging; DEGRADES the baseline — never score with this")
    sp.add_argument("--tp-size", type=int, default=None, help="tensor-parallel size (multi-GPU)")
    sp.add_argument("--moe-runner-backend", default=None,
                    help="sglang MoE backend (e.g. 'triton')")
    sp.add_argument("--candidate-moe-runner-backend", default=None,
                    help="candidate-only MoE backend override (framework-mode backend swaps)")
    sp.add_argument("--disable-custom-all-reduce", action="store_true",
                    help="needed for TP>2 over PCIe (no NVLink)")
    sp.add_argument("--candidate-disable-custom-all-reduce", action=argparse.BooleanOptionalAction, default=None,
                    help="candidate-only custom-all-reduce override")
    sp.add_argument("--engine-kwargs-json", default=None,
                    help="JSON object merged into both SGLang Engine kwargs")
    sp.add_argument("--candidate-engine-kwargs-json", default=None,
                    help="JSON object merged into candidate SGLang Engine kwargs")
    # optional: record the result into a commit-reveal ledger
    sp.add_argument("--ledger", default=None, help="ledger json to record the score into")
    sp.add_argument("--hotkey", default=None, help="miner hotkey (with --ledger)")
    sp.add_argument("--round", type=int, default=0, help="round id (with --ledger)")
    sp.add_argument("--framework-mode", action="store_true",
                    help="miner may patch the engine (setup()); gate on token-match vs the stock baseline, not in-process KL")
    sp.add_argument("--token-match-threshold", type=float, default=0.99,
                    help="framework-mode minimum token match fraction")
    sp.add_argument("--isolate", action="store_true",
                    help="run the candidate in a no-egress network namespace (auto-on with --framework-mode); needs root")
    sp.add_argument("--allow-unsafe-no-isolation", action="store_true",
                    help="DEV ONLY: continue if candidate no-egress isolation is unavailable")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("bench",
                        help="realistic eval: throughput on real benchmark prompts, gated by task accuracy + KL")
    sp.add_argument("bundle")
    sp.add_argument("--model", required=True)
    sp.add_argument("--benchmarks", default="gsm8k",
                    help="comma-separated: gsm8k, mmlu, long_math")
    sp.add_argument("--samples", type=int, default=32, help="problems per benchmark")
    sp.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    sp.add_argument("--max-new-tokens", type=int, default=None,
                    help="override the benchmark decode budget")
    sp.add_argument("--timed-iters", type=int, default=2)
    sp.add_argument("--prompt-seed", type=int, default=0)
    sp.add_argument("--acc-tolerance", type=float, default=0.02)
    sp.add_argument("--kl-threshold", type=float, default=5e-3, help="dense KL gate on the benchmark prompts")
    sp.add_argument("--argmax-disagree-rate", type=float, default=0.01,
                    help="max fraction of positions whose top token may flip (sparse-cheat guard)")
    sp.add_argument("--p99-kl-threshold", type=float, default=None, help="optional p99 KL gate (catastrophic tail)")
    sp.add_argument("--kl-advisory", action="store_true",
                    help="report KL but don't gate on it (big MoE: noise-dominated; rely on accuracy)")
    sp.add_argument("--top-logprobs", type=int, default=20, help="top-k logprobs for the KL gate (0 disables)")
    sp.add_argument("--ignore-eos", action="store_true",
                    help="force generation to the max token budget; useful for long-decode throughput probes")
    sp.add_argument("--mem-fraction", type=float, default=0.6,
                    help="sglang mem_fraction_static (use ~0.9 for big models like gpt-oss-120b)")
    sp.add_argument("--no-deterministic", action="store_true")
    sp.add_argument("--attention-backend", default=None,
                    help="sglang attention backend (default: auto-pick best per-HW, e.g. fa3/flashinfer)")
    sp.add_argument("--candidate-attention-backend", default=None,
                    help="candidate-only attention backend override")
    sp.add_argument("--disable-cuda-graph", action="store_true",
                    help="eager mode for quick debugging; DEGRADES the baseline — never score with this")
    sp.add_argument("--tp-size", type=int, default=None, help="tensor-parallel size (multi-GPU)")
    sp.add_argument("--moe-runner-backend", default=None,
                    help="sglang MoE backend (e.g. 'triton')")
    sp.add_argument("--candidate-moe-runner-backend", default=None,
                    help="candidate-only MoE backend override (framework-mode backend swaps)")
    sp.add_argument("--disable-custom-all-reduce", action="store_true",
                    help="needed for TP>2 over PCIe (no NVLink)")
    sp.add_argument("--candidate-disable-custom-all-reduce", action=argparse.BooleanOptionalAction, default=None,
                    help="candidate-only custom-all-reduce override")
    sp.add_argument("--engine-kwargs-json", default=None,
                    help="JSON object merged into both SGLang Engine kwargs")
    sp.add_argument("--candidate-engine-kwargs-json", default=None,
                    help="JSON object merged into candidate SGLang Engine kwargs")
    sp.add_argument("--framework-mode", action="store_true",
                    help="miner may patch/swap the engine; gate on token-match vs the stock baseline, not in-process KL")
    sp.add_argument("--token-match-threshold", type=float, default=0.99,
                    help="framework-mode minimum token match fraction")
    sp.add_argument("--isolate", action="store_true",
                    help="run the candidate in a no-egress network namespace (auto-on with --framework-mode); needs root")
    sp.add_argument("--allow-unsafe-no-isolation", action="store_true",
                    help="DEV ONLY: continue if candidate no-egress isolation is unavailable")
    sp.add_argument("--ledger", default=None)
    sp.add_argument("--hotkey", default=None)
    sp.add_argument("--round", type=int, default=0)
    sp.set_defaults(func=cmd_bench)

    # ---- commit-reveal / scoring ledger ----
    sp = sub.add_parser("hash", help="print a bundle's deterministic content hash")
    sp.add_argument("bundle")
    sp.set_defaults(func=cmd_hash)

    sp = sub.add_parser("commit", help="post a commitment for a bundle (commit phase)")
    sp.add_argument("bundle")
    sp.add_argument("--hotkey", required=True)
    sp.add_argument("--salt", required=True)
    sp.add_argument("--round", type=int, default=0)
    sp.add_argument("--ledger", default="optima_ledger.json")
    sp.set_defaults(func=cmd_commit)

    sp = sub.add_parser("reveal", help="reveal a previously committed bundle (reveal phase)")
    sp.add_argument("bundle")
    sp.add_argument("--hotkey", required=True)
    sp.add_argument("--salt", required=True)
    sp.add_argument("--round", type=int, default=0)
    sp.add_argument("--ledger", default="optima_ledger.json")
    sp.set_defaults(func=cmd_reveal)

    sp = sub.add_parser("ledger", help="show ledger state (champion, counts)")
    sp.add_argument("--ledger", default="optima_ledger.json")
    sp.set_defaults(func=cmd_ledger)

    sp = sub.add_parser("settle", help="settle a round: king-of-the-hill + weights")
    sp.add_argument("--round", type=int, default=0)
    sp.add_argument("--margin", type=float, default=0.02)
    sp.add_argument("--ledger", default="optima_ledger.json")
    sp.set_defaults(func=cmd_settle)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
