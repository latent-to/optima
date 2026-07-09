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

from optima.manifest import (all_declared_cuda_sources, all_declared_dep_patches,
                             load_manifest, resolve_source)
from optima.sandbox import scan_path


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
    # THE emission-policy seam: the ledger says who earns what; this command only
    # relays it. (Never re-derive winner-take-all here — see Ledger.current_weights.)
    weights = led.current_weights(per_slot=args.per_slot)
    if not weights:
        print(f"no champion(s) in {args.ledger}; nothing to weight")
        return 1
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


def cmd_chain_package(args: argparse.Namespace) -> int:
    from optima.chain.fetch import package_bundle

    out, ch = package_bundle(args.bundle, args.out)
    print(f"archive:      {out}")
    print(f"content_hash: {ch}")
    print("host the archive at a stable URL, then commit it: optima chain-submit "
          f"{args.bundle} --url <URL> --netuid <N> --network <WSS>")
    return 0


def cmd_chain_submit(args: argparse.Namespace) -> int:
    from optima.chain.payload import PayloadError
    from optima.chain.submit import submit_bundle

    from optima import chain

    subtensor = wallet = None
    if not args.dry_run:
        import bittensor as bt

        subtensor = chain.connect(args.network)
        wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
    try:
        res = submit_bundle(subtensor, wallet, args.netuid, args.bundle, args.url,
                            blocks_until_reveal=args.blocks_until_reveal,
                            dry_run=args.dry_run)
    except PayloadError as e:
        print(f"REFUSED before signing: {e}")
        return 2
    print(f"content_hash: {res['content_hash']}")
    print(f"payload:      {res['payload']}")
    if args.dry_run:
        print("DRY RUN — nothing sent. The payload above is what would be committed "
              f"(timelock, reveals after {args.blocks_until_reveal} blocks).")
        return 0
    ok = bool(res.get("submitted"))
    print(f"set_reveal_commitment submitted={ok} "
          f"(reveals after {args.blocks_until_reveal} blocks; the validator picks it "
          "up on its next pass after the reveal)")
    return 0 if ok else 1


def cmd_chain_status(args: argparse.Namespace) -> int:
    from optima import chain
    from optima.chain.payload import decode_payload

    subtensor = chain.connect(args.network)
    block = int(subtensor.get_current_block())
    print(f"network: {args.network}  netuid: {args.netuid}  block: {block}")
    mg = chain.fetch_metagraph(subtensor, args.netuid)
    print(f"neurons: {len(mg.uids)}  permits: {sum(mg.validator_permit)}")
    if args.wallet:
        import bittensor as bt

        wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
        hk = wallet.hotkey.ss58_address
        uid = mg.uid_of(hk)
        permit = bool(uid is not None and uid < len(mg.validator_permit)
                      and mg.validator_permit[uid])
        print(f"our hotkey {hk}: uid={uid} permit={permit}")
    revealed = chain.read_revealed_commitments(subtensor, args.netuid)
    print(f"revealed commitments: {len(revealed)}")
    for hk, rc in sorted(revealed.items(), key=lambda kv: kv[1].block):
        ref = decode_payload(hk, rc.block, rc.data)
        if ref is None:
            print(f"  block {rc.block}  {hk}  (unparseable payload)")
        else:
            print(f"  block {rc.block}  {hk}  {ref.content_hash[:16]}…  {ref.url}")
    return 0


def cmd_chain_validate(args: argparse.Namespace) -> int:
    from optima import chain
    from optima.chain.validator_loop import (
        command_evaluator,
        run_validator,
        verify_evaluator,
    )

    if args.eval_cmd:
        evaluator = command_evaluator(args.eval_cmd, timeout_s=args.eval_timeout)
    else:
        evaluator = verify_evaluator(device=args.eval_device, timeout_s=args.eval_timeout)
        print("NOTE: verify-mode evaluator (pass/fail plumbing score of 1.0) — a 1.0 "
              "never clears the dethrone margin, so crown plumbing runs with "
              "--margin 0; wire --eval-cmd to the full GPU gate chain for real scoring")
    subtensor = chain.connect(args.network, retry_forever=not args.once)
    wallet = None
    if not args.dry_run_weights:
        import bittensor as bt

        wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
    res = run_validator(subtensor, wallet, args.netuid, ledger_path=args.ledger,
                        bundles_dir=args.bundles_dir, evaluator=evaluator,
                        margin=args.margin, interval_s=args.interval, once=args.once,
                        dry_run_weights=args.dry_run_weights)
    if args.once and res is not None:
        print(f"pass @block {res.block} (round {res.round_id}): seen={res.seen} "
              f"new={len(res.new)} copies={len(res.copies)} rejected={len(res.rejected)}")
        for ch_, ok in res.evaluated.items():
            print(f"  evaluated {ch_[:16]}… passed={ok}")
        for ch_, why in res.rejected.items():
            print(f"  rejected  {ch_[:16]}… {why}")
        print(f"weights: {res.weights}  pushed={res.weights_pushed}")
    return 0


def cmd_chain_register(args: argparse.Namespace) -> int:
    import bittensor as bt

    from optima import chain

    subtensor = chain.connect(args.network)
    wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
    hk = wallet.hotkey.ss58_address
    if subtensor.is_hotkey_registered(hotkey_ss58=hk, netuid=args.netuid):
        print(f"already registered: {hk}")
    else:
        cost = subtensor.recycle(args.netuid)
        print(f"registering {hk} on netuid {args.netuid} (burn ≈ {cost}) …")
        resp = subtensor.burned_register(wallet, args.netuid)
        ok = bool(getattr(resp, "success", resp))
        print(f"burned_register success={ok} {getattr(resp, 'message', '')}")
        if not ok:
            return 1
    for check in chain.preflight(subtensor, wallet, args.netuid):
        print(f"  [{'ok' if check.ok else 'MISSING'}] {check.name}: {check.detail}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    from optima.sandbox import scan_tree

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
    # Recursive guard: catch a vendored/extra .py the per-op (entry-only) scan misses, and
    # (fail-closed, manifest now loaded) any file that's neither a scanned .py, a declared
    # cuda_source, nor benign metadata — e.g. an undeclared .cu or a stray .so.
    # Exact file match, not startswith: a prefix filter would also drop violations in
    # e.g. "kernels/silu.py_evil.py" because it string-prefixes "kernels/silu.py".
    op_sources = {op.source for op in m.ops}
    declared_cuda = all_declared_cuda_sources(args.bundle, m)
    declared_patches = all_declared_dep_patches(args.bundle, m)
    extra = [v for v in scan_tree(args.bundle, declared_cuda_sources=declared_cuda,
                                  declared_dep_patches=declared_patches).violations
             if v.split(":", 1)[0] not in op_sources]
    if extra:
        print("  [VIOLATIONS] vendored/extra/undeclared files (recursive scan):")
        for v in extra:
            print(f"      {v}")
        rc = 2
    return rc


def _recursive_scan_ok(bundle: str, manifest=None) -> bool:
    """Fail-closed vendored-tree guard for the eval paths: scan every bundle .py, not just the
    declared entries (a vendored library .py using open/importlib/subprocess must not slip in
    unscanned). Prints violations; returns False if any.

    ``manifest`` (already loaded by the caller) supplies the declared ``cuda_sources``
    allowlist, so scan_tree runs in its fail-closed mode: any file that's neither a
    scanned ``.py``, a declared cuda_source, nor benign metadata is rejected. Passing
    ``None`` falls back to the old (looser) behavior — kept only for callers that scan
    without a manifest; every call site in this file now has one available.
    """
    from optima.sandbox import scan_tree

    declared_cuda = all_declared_cuda_sources(bundle, manifest) if manifest is not None else None
    declared_patches = (all_declared_dep_patches(bundle, manifest)
                        if manifest is not None else None)
    tree = scan_tree(bundle, declared_cuda_sources=declared_cuda,
                     declared_dep_patches=declared_patches)
    if not tree.ok:
        print("  [FAIL] recursive policy scan (vendored-tree guard):")
        for v in tree.violations:
            print(f"      {v}")
    return tree.ok


def _declared_model(bundle: str, op) -> str | None:
    """Dev convenience: read the model an op's metadata JSON declares, to pick the
    validator's per-model slot profile when --model isn't given. Never reads thresholds
    from metadata (those are validator-owned in slots.MODEL_PROFILES) — only the model id,
    which selects WHICH validator profile applies. Best-effort; returns None on any issue."""
    if not getattr(op, "metadata", None):
        return None
    try:
        import json
        from pathlib import Path

        meta = json.loads((Path(bundle) / op.metadata).read_text())
        return meta.get("model") or meta.get("model_profile")
    except Exception:
        return None


def cmd_verify(args: argparse.Namespace) -> int:
    from optima.slots import SLOTS, get_slot, model_profile, slot_for_model
    from optima.verify import format_verify, verify_entry

    m = load_manifest(args.bundle)
    if not _recursive_scan_ok(args.bundle, manifest=m):  # vendored-tree guard (every .py, not just entries)
        return 2
    import torch
    # Mirror the ACTUAL device resolution, including verify_collective's fallback:
    # a collective needs world_size GPUs, so a 1-GPU box silently runs gloo/CPU.
    ws = getattr(args, "world_size", None) or 2
    has_collective = any(op.slot in SLOTS and get_slot(op.slot).kind == "collective"
                         for op in m.ops)
    cuda_ok = torch.cuda.is_available() and (
        not has_collective or torch.cuda.device_count() >= ws)
    if (args.device or ("cuda" if cuda_ok else "cpu")) == "cpu":
        print("[note] some or all of this verify runs on CPU: it checks op-correctness "
              "only — it does not predict GPU throughput, CUDA-graph capture, or the "
              "fidelity gates (see docs/GPU_SETUP.md).")
    rc = 0
    for op in m.ops:
        if op.slot not in SLOTS:
            print(f"  [SKIP] {op.slot}: not a known slot on this validator")
            continue
        model_key = args.model or _declared_model(args.bundle, op)
        if model_profile(model_key, op.slot) is not None:
            via = "via --model" if args.model else "declared in metadata"
            print(f"  [profile] {op.slot}: model {model_key!r} ({via}) -> validator slot profile "
                  "(activation + low-bit metric)")
        slot = slot_for_model(op.slot, model_key)
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
            result = verify_collective(slot, str(src), op.entry, prepare_name=op.prepare,
                                       world_size=ws, device=args.device, seed=args.seed,
                                       jitter_seed=args.seed,  # anti shape-branch, like per-op
                                       model_key=model_key,
                                       # rebuild plan (declared cuda_sources) must apply
                                       # in the ranks that load the kernel
                                       bundle_path=str(args.bundle))
            print(format_verify(result))
            if not result.passed:
                rc = 2
            continue

        # Load + run the miner kernel in a FRESH spawned process, so THIS trusted CLI
        # process never imports miner code (no in-process RCE sink). Production must also
        # namespace/no-egress that child; this removes the trusted-process execution.
        from optima.eval._launch import call_in_subprocess
        from optima.verify import verify_entry_from_source

        result = call_in_subprocess(
            verify_entry_from_source, op.slot, str(src), op.entry,
            prepare_name=op.prepare, dtype_name=args.dtype, device=args.device, seed=args.seed,
            jitter_seed=args.seed,  # count-dim jitter so shapes vary per run (anti shape-branch)
            model_key=model_key,  # validator per-model slot profile (activation + metric)
            override_point=op.override_point,  # compose a miner epilogue into the base kernel
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
    if not _recursive_scan_ok(args.bundle, manifest=m):  # vendored-tree guard (every .py, not just entries)
        return 2
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

    # Per-slot calibrated KL threshold (e.g. attention's higher floor) overrides the generic
    # default unless KL is advisory; the user's explicit --kl-threshold still applies as the
    # fallback for slots without a calibrated value.
    from optima.slots import get_slot as _get_slot
    _slot_kl = _get_slot(m.ops[0].slot).kl_threshold
    _kl_threshold = None if args.kl_advisory else (_slot_kl if _slot_kl is not None else args.kl_threshold)

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
        warmup_iters=args.warmup_iters,
        speedup_margin=args.speedup_margin,
        prompt_seed=args.prompt_seed,
        top_logprobs_num=args.top_logprobs,
        ignore_eos=args.ignore_eos,
        kl_threshold=_kl_threshold,
        argmax_disagree_rate_threshold=args.argmax_disagree_rate,
        p99_kl_threshold=args.p99_kl_threshold,
        deterministic=not args.no_deterministic,
        attention_backend=args.attention_backend,
        disable_cuda_graph=args.disable_cuda_graph,
        mem_fraction_static=args.mem_fraction,
        tp_size=args.tp_size,
        max_running_requests=args.max_running_requests,
        moe_runner_backend=args.moe_runner_backend,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        candidate_attention_backend=args.candidate_attention_backend,
        candidate_moe_runner_backend=args.candidate_moe_runner_backend,
        candidate_disable_custom_all_reduce=args.candidate_disable_custom_all_reduce,
        extra_engine_kwargs=_json_obj(args.engine_kwargs_json),
        candidate_extra_engine_kwargs=_json_obj(args.candidate_engine_kwargs_json),
        fidelity_mode=args.fidelity_mode,
        audit_rate=args.audit_rate,
    )
    if _slot_kl is not None and not args.kl_advisory and cfg.fidelity_mode == "kl":
        print(f"  (using {m.ops[0].slot}'s calibrated KL threshold {_slot_kl:g})")
    if cfg.fidelity_mode == "audit":
        print(f"  (fidelity=audit: extra untimed quality launch, in-engine audit at "
              f"rate {cfg.audit_rate:g}, KL advisory)")
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
    if report.baseline2 is not None:
        b2 = report.baseline2
        print(f"baseline'  {b2.tok_per_s:8.1f} tok/s  (trailing bookend; baseline noise {report.noise:.1%})")
    if not report.confident:
        verdict = "NO-DECISION (box too noisy / un-bracketed; re-queue, never crown)"
    elif report.passed_speedup:
        verdict = "PASS (noise-confident real win)"
    else:
        verdict = "below the noise-derived bar"
    print(f"speedup    {report.speedup:8.3f}x  (needs >= {report.required_speedup:.3f} = "
          f"1 + max({cfg.speedup_margin:g}, {cfg.score_k:g}*noise) -> {verdict})")
    if report.fidelity_mode == "audit":
        print(f"quality    in-engine audit: {report.audit_desc} -> "
              f"{'PASS' if report.passed_quality else 'FAIL'}")
        print(f"           KL (ADVISORY, not gated — launch-nondeterminism confounded): "
              f"mean_kl={report.kl.mean_kl:.3e} max_kl={report.kl.max_kl:.3e} "
              f"argmax_disagree={report.kl.argmax_disagreements}/{report.kl.num_positions} "
              f"token_match={report.token_match:.4f}")
    else:
        print(f"quality    mean_kl={report.kl.mean_kl:.3e} max_kl={report.kl.max_kl:.3e} "
              f"argmax_disagree={report.kl.argmax_disagreements}/{report.kl.num_positions}  "
              f"token_match={report.token_match:.4f}{' (GATE)' if cfg.framework_mode else ''} -> "
              f"{'PASS' if report.passed_quality else 'FAIL'}")
    print(f"SCORE      {report.score:.3f}  (crownable speedup, else 0.0)")

    if getattr(args, "ledger", None) and getattr(args, "hotkey", None):
        from optima.bundle_hash import content_hash
        from optima.commit_reveal import Ledger
        from optima.compat import PINNED_SGLANG

        ch = content_hash(args.bundle)
        led = Ledger.load(args.ledger)
        led.record_score(args.hotkey, ch, args.round, report.score, report.kl.mean_kl,
                         report.passed_quality, sglang_version=PINNED_SGLANG, slot=m.ops[0].slot)
        led.save(args.ledger)
        print(f"recorded -> {args.ledger} (hotkey={args.hotkey}, round={args.round}, "
              f"slot={m.ops[0].slot}, sglang={PINNED_SGLANG})")
    return 0 if report.passed_quality else 3


def cmd_bench(args: argparse.Namespace) -> int:
    from optima.slots import SLOTS
    from optima.eval.capability import evaluate_capability
    from optima.eval.throughput_kl import EvalConfig

    m = load_manifest(args.bundle)
    if not _recursive_scan_ok(args.bundle, manifest=m):  # vendored-tree guard (every .py, not just entries)
        return 2
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

    from optima.slots import get_slot as _get_slot
    _slot_kl = _get_slot(m.ops[0].slot).kl_threshold
    _kl_threshold = None if args.kl_advisory else (_slot_kl if _slot_kl is not None else args.kl_threshold)
    if args.samples < 100 and not args.kl_advisory:
        print(f"  [note] --samples {args.samples} is small for the accuracy gate "
              "(~12% std at n=12); KL is the primary gate, use ~100-200 for a real accuracy floor.")

    cfg = EvalConfig(
        model_path=args.model,
        dtype=args.dtype,
        timed_iters=args.timed_iters,
        prompt_seed=args.prompt_seed,
        top_logprobs_num=args.top_logprobs,
        ignore_eos=args.ignore_eos,
        kl_threshold=_kl_threshold,
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
        max_running_requests=args.max_running_requests,
        moe_runner_backend=args.moe_runner_backend,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        candidate_attention_backend=args.candidate_attention_backend,
        candidate_moe_runner_backend=args.candidate_moe_runner_backend,
        candidate_disable_custom_all_reduce=args.candidate_disable_custom_all_reduce,
        extra_engine_kwargs=_json_obj(args.engine_kwargs_json),
        candidate_extra_engine_kwargs=_json_obj(args.candidate_engine_kwargs_json),
    )
    names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    # (bench keeps the EvalConfig default warmup; its tok/s is documented noise)
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
    b2 = f"  baseline' {report.baseline2_tok_s:8.1f}" if report.baseline2_tok_s > 0 else ""
    print(f"throughput baseline {report.baseline_tok_s:8.1f} tok/s  candidate {report.candidate_tok_s:8.1f} tok/s{b2}")
    if not report.confident:
        sp_verdict = "NO-DECISION (box too noisy; re-queue)"
    elif report.passed_speedup:
        sp_verdict = "PASS (noise-confident)"
    else:
        sp_verdict = "below the noise-derived bar"
    print(f"speedup    {report.speedup:8.3f}x  (needs >= {report.required_speedup:.3f}, "
          f"baseline noise {report.noise:.1%}) -> {sp_verdict}")
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

        from optima.compat import PINNED_SGLANG

        ch = content_hash(args.bundle)
        led = Ledger.load(args.ledger)
        led.record_score(args.hotkey, ch, args.round, report.score, report.kl.mean_kl,
                         report.passed_quality, sglang_version=PINNED_SGLANG, slot=m.ops[0].slot)
        led.save(args.ledger)
        print(f"recorded -> {args.ledger} (hotkey={args.hotkey}, round={args.round}, "
              f"slot={m.ops[0].slot}, sglang={PINNED_SGLANG})")
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
    from optima.copy_fingerprint import (
        bundle_fingerprint,
        bundle_slot_file_fingerprints,
        bundle_slot_fingerprints,
        bundle_structural_fingerprint,
    )

    ch = content_hash(args.bundle)
    fp = bundle_fingerprint(args.bundle)  # reformat-invariant near-copy signal (auto-demotes)
    slot_fps = bundle_slot_fingerprints(args.bundle)  # per-slot: a padded bundle can't hide a stolen slot
    file_fps = bundle_slot_file_fingerprints(args.bundle)  # per-file: nor a RELOCATED stolen body
    sfp = bundle_structural_fingerprint(args.bundle)  # rename/constant-tweak skeleton (advisory)
    led = Ledger.load(args.ledger)
    # Query advisory structural matches BEFORE recording this reveal (so we don't match self).
    advisory = led.structural_near_copies(sfp, args.hotkey)
    try:
        rev = led.reveal(args.hotkey, ch, args.salt, args.round, fingerprint=fp,
                         structural_fingerprint=sfp, slot_fingerprints=slot_fps,
                         slot_file_fingerprints=file_fps)
    except RevealError as e:
        print(f"REJECTED: {e}")
        return 2
    led.save(args.ledger)
    print(f"revealed hotkey={args.hotkey} content={ch[:16]}... original={rev.original}")
    if not rev.original:
        print("  -> flagged as a COPY (an earlier commit to this exact content, its "
              "reformatted-but-identical structure, or a bundle whose kernel source "
              "this one contains exists); earns 0")
    elif advisory:
        print(f"  ⚠ ADVISORY: structurally similar to earlier submission(s) by {', '.join(advisory)} "
              "(possible rename/constant-tweak copy) — flagged for review, not auto-demoted")
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
    from optima.compat import PINNED_SGLANG

    led = Ledger.load(args.ledger)
    if getattr(args, "per_slot", False):
        res = led.settle_per_slot(args.round, margin=args.margin, current_sglang_version=PINNED_SGLANG)
        led.save(args.ledger)
        print("per-slot championships (emission split across slots):")
        for slot, champ in sorted(res.champions.items()):
            changed = " (NEW)" if res.title_changes.get(slot) else ""
            stale = "  ⚠ STALE pin — re-baseline" if slot in res.stale_slots else ""
            print(f"  {slot or '(unlabeled)':32s} {champ.hotkey} score={champ.score:.3f}{changed}{stale}")
        if res.rejected_copies:
            print(f"rejected copies: {', '.join(res.rejected_copies)}")
        print(f"weights: {res.weights}")
        return 0
    res = led.settle(args.round, margin=args.margin, current_sglang_version=PINNED_SGLANG)
    led.save(args.ledger)
    print(f"title_changed={res.title_changed} challenger_score={res.challenger_score:.3f}")
    if res.champion:
        print(f"champion: {res.champion.hotkey} score={res.champion.score:.3f}")
    if res.champion_stale:
        print(f"  ⚠ champion was crowned under a DIFFERENT sglang pin than {PINNED_SGLANG}; "
              "re-baseline it (re-evaluate the champion bundle on the current pin).")
    if res.rejected_copies:
        print(f"rejected copies: {', '.join(res.rejected_copies)}")
    print(f"weights: {res.weights}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="optima",
        description=(
            "Optima validator harness.\n"
            "\n"
            "Commands by workflow:\n"
            "  develop a kernel (miner) ... slots, scan, verify, evaluate, bench\n"
            "  submit on-chain (miner) .... hash, chain-register, chain-package,\n"
            "                               chain-submit, chain-status\n"
            "  score + settle (validator) . chain-validate, settle, ledger, set-weights,\n"
            "                               commit, reveal (local-ledger simulation)\n"
            "  environment checks ......... compat, chain-compat\n"
            "\n"
            "New to Optima? Start with docs/MINER_GUIDE.md."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    sp.add_argument("--network", default="finney",
                    help="named network or an explicit wss:// endpoint URL")
    sp.add_argument("--wallet", default="default")
    sp.add_argument("--hotkey", default="default")
    sp.add_argument("--per-slot", action="store_true",
                    help="weights from the per-slot championships (emission split), "
                         "matching settle --per-slot")
    sp.add_argument("--dry-run", action="store_true",
                    help="build + print the (uids, weights) payload, do NOT submit")
    sp.set_defaults(func=cmd_set_weights)

    # ---- chain: miner submission + the validator loop ----
    sp = sub.add_parser("chain-package",
                        help="tar.gz a bundle for hosting; prints the content hash to commit")
    sp.add_argument("bundle")
    sp.add_argument("--out", default=None, help="archive path (default <bundle>.tar.gz)")
    sp.set_defaults(func=cmd_chain_package)

    sp = sub.add_parser("chain-submit",
                        help="miner: commit a bundle (hash + fetch URL) on-chain via "
                             "timelock commit-reveal")
    sp.add_argument("bundle")
    sp.add_argument("--url", required=True, help="where the validator fetches the tar.gz")
    sp.add_argument("--netuid", type=int, required=True)
    sp.add_argument("--network", required=True,
                    help="named network or an explicit wss:// endpoint URL")
    sp.add_argument("--wallet", default="default")
    sp.add_argument("--hotkey", default="default", help="the MINER hotkey name")
    sp.add_argument("--blocks-until-reveal", type=int, default=10,
                    help="timelock length; the payload is unreadable until then")
    sp.add_argument("--dry-run", action="store_true",
                    help="build + print the payload, do NOT sign or submit")
    sp.set_defaults(func=cmd_chain_submit)

    sp = sub.add_parser("chain-status",
                        help="subnet snapshot: block, neurons, permits, revealed submissions")
    sp.add_argument("--netuid", type=int, required=True)
    sp.add_argument("--network", required=True)
    sp.add_argument("--wallet", default=None, help="also report this wallet's uid/permit")
    sp.add_argument("--hotkey", default="default")
    sp.set_defaults(func=cmd_chain_status)

    sp = sub.add_parser("chain-validate",
                        help="the validator loop: commitments -> fetch -> evaluate -> "
                             "settle -> weights")
    sp.add_argument("--netuid", type=int, required=True)
    sp.add_argument("--network", required=True)
    sp.add_argument("--wallet", default="default")
    sp.add_argument("--hotkey", default="default", help="the VALIDATOR hotkey name")
    sp.add_argument("--ledger", default="chain_ledger.json")
    sp.add_argument("--bundles-dir", default="chain_bundles",
                    help="where fetched submissions are cached (keyed by content hash)")
    sp.add_argument("--eval-cmd", default=None,
                    help="eval command template with {bundle} and {report} placeholders "
                         "(exit 0 = passed; JSON report carries score/kl_mean/slot). "
                         "Default: verify-mode plumbing evaluator (CPU pass/fail)")
    sp.add_argument("--eval-device", default="cpu",
                    help="verify-mode device (default cpu)")
    sp.add_argument("--eval-timeout", type=float, default=3600.0)
    sp.add_argument("--margin", type=float, default=0.02, help="settle dethrone margin")
    sp.add_argument("--interval", type=float, default=60.0, help="seconds between passes")
    sp.add_argument("--once", action="store_true", help="single pass, then exit")
    sp.add_argument("--dry-run-weights", action="store_true",
                    help="run the full loop but never submit weights")
    sp.set_defaults(func=cmd_chain_validate)

    sp = sub.add_parser("chain-register",
                        help="register this hotkey on a subnet (burned_register; needs "
                             "the coldkey password) + preflight")
    sp.add_argument("--netuid", type=int, required=True)
    sp.add_argument("--network", required=True)
    sp.add_argument("--wallet", default="default")
    sp.add_argument("--hotkey", default="default")
    sp.set_defaults(func=cmd_chain_register)

    sp = sub.add_parser("scan", help="static policy scan of a bundle")
    sp.add_argument("bundle")
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser(
        "verify", help="op-level correctness vs reference",
        epilog=("examples:\n"
                "  # CPU dry-run (no GPU needed; the miner-guide inner loop)\n"
                "  optima verify examples/miner_silu_torch --device cpu --dtype float32\n"
                "  # real shapes/dtypes on a GPU box\n"
                "  optima verify my_bundle --device cuda --dtype bfloat16\n"
                "  # a collective slot at the arena's TP size\n"
                "  optima verify my_bundle --device cuda --world-size 4"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sp.add_argument("bundle")
    sp.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    sp.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--world-size", type=int, default=None, dest="world_size",
                    help="ranks for DISTRIBUTED verify of collective slots (default 2; "
                         "use the arena TP size, e.g. 4, on a multi-GPU box)")
    sp.add_argument("--model", default=None,
                    help="validator model key for the per-model slot profile (activation + "
                         "low-bit metric), e.g. MiniMax-M3. Default: the model declared in the "
                         "op's metadata (dev convenience); production uses the served-model key.")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser(
        "evaluate", help="end-to-end throughput + KL on a model",
        epilog=("examples (always launch via `python -m optima.cli` on GPU —\n"
                "sglang spawns the scheduler with mp spawn):\n"
                "  # quick smoke on a small model\n"
                "  python -m optima.cli evaluate my_bundle --model Qwen/Qwen2.5-1.5B-Instruct \\\n"
                "      --num-prompts 64 --max-new-tokens 64\n"
                "  # nondeterministic arena: fidelity via the in-engine audit, KL advisory\n"
                "  python -m optima.cli evaluate my_bundle --model <model> \\\n"
                "      --fidelity-mode audit --no-deterministic"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sp.add_argument("bundle")
    sp.add_argument("--model", required=True, help="model path for sglang.Engine")
    sp.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    sp.add_argument("--max-new-tokens", type=int, default=64)
    sp.add_argument("--num-prompts", type=int, default=32)
    sp.add_argument("--timed-iters", type=int, default=3, help="median-of-K timed passes per launch")
    sp.add_argument("--warmup-iters", type=int, default=2,
                    help="untimed heat-soak rounds before the first timed pass. On boxes where "
                         "clock-locking is unavailable (tenant pods), thermal ramp lands in the "
                         "B/B' bookends as baseline 'noise' and inflates the crowning bar — raise "
                         "this until the bookends agree instead")
    sp.add_argument("--speedup-margin", type=float, default=0.005,
                    help="FLOOR on the required improvement; the actual bar is "
                         "1 + max(margin, 2*measured_noise). Keep low — real wins stack at 1-2%%; "
                         "the noise term, not this floor, guards an unstable box")
    sp.add_argument("--prompt-seed", type=int, default=0, help="per-epoch prompt sampling seed")
    sp.add_argument("--top-logprobs", type=int, default=20)
    sp.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True,
                    help="force generation to the max token budget so baseline and candidate emit IDENTICAL "
                         "token counts (pure latency comparison, no EOS-timing gaming). ON for scoring; "
                         "--no-ignore-eos only for a natural-length probe")
    sp.add_argument("--kl-threshold", type=float, default=5e-3)
    sp.add_argument("--argmax-disagree-rate", type=float, default=0.01,
                    help="max fraction of positions whose top token may flip (sparse-cheat guard)")
    sp.add_argument("--p99-kl-threshold", type=float, default=None, help="optional p99 KL gate (catastrophic tail)")
    sp.add_argument("--kl-advisory", action="store_true", help="report KL but don't gate on it")
    sp.add_argument("--fidelity-mode", choices=("kl", "audit"), default="kl",
                    help="quality gate: 'kl' = rollout-KL vs the baseline launch (valid only on a "
                         "deterministic-capable arena); 'audit' = in-engine per-call comparison vs "
                         "the stock baseline under the slot's verify tolerances (extra untimed "
                         "quality launch; KL becomes advisory). Use 'audit' where two identical "
                         "launches aren't logit-identical (measured 2026-07-07: bit-stock "
                         "candidates scored mean_kl 0.8-0.96 on eager fa4/NVFP4).")
    sp.add_argument("--audit-rate", type=float, default=0.05,
                    help="fidelity-mode=audit: fraction of eligible dispatcher calls audited")
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
    sp.add_argument("--max-running-requests", type=int, default=None,
                    help="cap concurrent running requests = score at a serving-realistic batch (report M2)")
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

    sp = sub.add_parser(
        "bench",
        help="realistic eval: throughput on real benchmark prompts, gated by task accuracy + KL",
        epilog=("examples:\n"
                "  # capability floor on a real task (start small, then raise --samples)\n"
                "  python -m optima.cli bench my_bundle --model Qwen/Qwen2.5-1.5B-Instruct \\\n"
                "      --benchmarks gsm8k --samples 128"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
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
    sp.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True,
                    help="force generation to the max token budget so baseline and candidate emit IDENTICAL "
                         "token counts (pure latency comparison, no EOS-timing gaming). ON for scoring; "
                         "--no-ignore-eos only for a natural-length probe")
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
    sp.add_argument("--max-running-requests", type=int, default=None,
                    help="cap concurrent running requests = score at a serving-realistic batch (report M2)")
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
    sp.add_argument("--per-slot", action="store_true",
                    help="per-slot championships (one champion per slot, emission split) — pays "
                         "specialists, vs the winner-take-all default")
    sp.set_defaults(func=cmd_settle)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
