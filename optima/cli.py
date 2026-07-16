"""Optima validator CLI — drives the submission pipeline end to end.

    python -m optima.cli slots
    python -m optima.cli scan      <bundle>
    python -m optima.cli verify    <bundle> [--dtype bfloat16] [--device cuda]

Pipeline (mirrors the validator flow):

    manifest -> static scan -> (isolated) load -> op-correctness -> register
             -> chain intake -> qualification (B/C/B'/T) -> settlement

SECURITY NOTE: ``verify`` imports the miner module, which runs
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


def cmd_model_provision(args: argparse.Namespace) -> int:
    from optima.model_provision import provision_model

    result = provision_model(
        args.model_root,
        args.publication_root,
        expected_content_digest=args.expected_content_digest,
        workers=args.workers,
    )
    print(json.dumps(
        {
            "content_digest": result.receipt.content_digest,
            "receipt_digest": result.receipt.receipt_digest,
            "receipt_path": str(result.receipt_path),
        },
        sort_keys=True,
    ))
    return 0


def cmd_release_verify(args: argparse.Namespace) -> int:
    from optima.release import reopen_release

    release = reopen_release(
        args.release_root,
        expected_descriptor_digest=args.descriptor_digest,
        expected_public_key=args.expected_public_key,
    )
    print(json.dumps(
        {
            "descriptor_digest": release.descriptor.digest,
            "engine_tree_digest": release.descriptor.engine_tree_digest,
            "public_key": release.signature.public_key,
            "release_tree_digest": release.release_tree_digest,
        },
        sort_keys=True,
    ))
    return 0


def cmd_release_context(args: argparse.Namespace) -> int:
    from optima.release import container_context, reopen_release

    release = reopen_release(
        args.release_root,
        expected_descriptor_digest=args.descriptor_digest,
        expected_public_key=args.expected_public_key,
    )
    result = container_context(
        release, args.destination, expected_public_key=args.expected_public_key
    )
    print(result)
    return 0


def cmd_set_weights(args: argparse.Namespace) -> int:
    from optima import chain
    from optima.chain.intake import (
        FinalizedIntakeStore,
        IntakeScope,
        SQLiteWeightPublicationJournal,
    )
    from optima.chain.weights import (
        reconcile_weight_publication,
        release_weight_publication_hold,
        resume_weight_projection,
    )
    from optima.economics import (
        EmissionsPolicyManifest,
        GlobalRewardProjectionContext,
        MetagraphMember,
    )
    from optima.target_catalog import default_target_catalog

    subtensor = chain.connect(args.network)
    import bittensor as bt

    wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
    validator_hotkey = wallet.hotkey.ss58_address
    scope = IntakeScope(str(subtensor.get_block_hash(0)).lower(), args.netuid)
    metagraph = chain.fetch_metagraph(subtensor, args.netuid)
    context = GlobalRewardProjectionContext(
        scope.digest,
        validator_hotkey,
        metagraph.block,
        metagraph.block_hash.lower(),
        tuple(
            MetagraphMember(uid, hotkey)
            for uid, hotkey in zip(metagraph.uids, metagraph.hotkeys, strict=True)
        ),
    )
    policy = EmissionsPolicyManifest(
        args.half_life_blocks,
        args.discovery_lifetime_blocks,
        args.discovery_pool_ppm,
    )
    catalog = default_target_catalog()
    with FinalizedIntakeStore(args.intake_db, scope=scope) as store:
        states = store.evaluation_stacks()
        catalogs = {state.arena_digest: catalog for state in states}
        projection = store.build_weight_projection(
            policy=policy,
            context=context,
            catalogs=catalogs,
            netuid=args.netuid,
        )
        journal = SQLiteWeightPublicationJournal(store, projection)
        if args.release_hold:
            if args.dry_run:
                raise SystemExit("--release-hold cannot be combined with --dry-run")
            released = release_weight_publication_hold(
                journal, reason=args.release_hold
            )
            print(
                f"released held weight publication {released.projection_digest}; "
                "run set-weights again to refresh and reconcile"
            )
            return 0
        if not args.dry_run:
            projection = resume_weight_projection(projection, journal)
            journal = SQLiteWeightPublicationJournal(store, projection)
        result = reconcile_weight_publication(
            subtensor,
            None if args.dry_run else wallet,
            projection,
            journal,
            refresh_blocks=args.refresh_blocks,
            dry_run=args.dry_run,
        )
    print(
        f"weight projection={projection.digest} status={result.status} "
        f"chain_matches={result.chain_matches} submitted={result.submitted}"
    )
    return 0 if result.status in {"dry_run", "confirmed", "pending"} else 2


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


def cmd_chain_validate(
    args: argparse.Namespace, *, arena_registry=None
) -> int:
    import logging

    from optima import chain
    from optima.chain.validator_loop import run_validator

    from optima.arena_service import ArenaServiceRegistry

    injected = arena_registry
    if not args.intake_only and (
        type(injected) is not ArenaServiceRegistry
        or not getattr(args, "arena_id", None)
    ):
        raise SystemExit(
            "chain-validate requires --intake-only or a validator-injected "
            "ArenaServiceRegistry plus --arena-id"
        )
    subtensor = chain.connect(args.network, retry_forever=not args.once)
    # Daemon-mode observability: between passes the loop reports only through the
    # "optima.chain.*" loggers (--once prints its own summary below). This must run
    # AFTER connect(): the bittensor import reconfigures global logging — it sets
    # every pre-existing third-party logger's level to CRITICAL (measured in the
    # 2026-07-10 soak: the ledger advanced every pass while the log stayed empty;
    # optima.chain.validator read level=50). Own the subtree outright: reset levels
    # to inherit, dedicated handler, no propagation upward.
    for _name, _lg in list(logging.root.manager.loggerDict.items()):
        if _name.startswith("optima.") and isinstance(_lg, logging.Logger):
            _lg.disabled = False
            _lg.setLevel(logging.NOTSET)
    _chain_lg = logging.getLogger("optima.chain")
    _chain_lg.setLevel(logging.INFO)
    _chain_lg.propagate = False
    if not _chain_lg.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        _chain_lg.addHandler(_handler)
    res = run_validator(
        subtensor,
        args.netuid,
        intake_db=args.intake_db,
        private_root=args.private_root,
        publication_root=args.publication_root,
        arena_registry=injected,
        arena_id=None if args.intake_only else args.arena_id,
        intake_only=args.intake_only,
        interval_s=args.interval,
        once=args.once,
    )
    if args.once and res is not None:
        print(
            f"intake @finalized {res.finalized_block}: seen={res.seen} "
            f"reserved={len(res.reserved)} published={len(res.published)} "
            f"copies={len(res.copies)} rejected={len(res.rejected)} "
            f"screens={len(res.screens)} decisions={len(res.decisions)} "
            f"settlements={len(res.settlements)} held={len(res.held)}"
        )
        for reservation, why in res.rejected.items():
            print(f"  rejected {reservation[:16]}… {why}")
        if args.intake_only:
            print("qualification/settlement: disabled by --intake-only")
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


def _declared_metadata(bundle: str, op) -> dict:
    """Read normative eligibility metadata; malformed content fails closed."""
    if not getattr(op, "metadata", None):
        return {}
    import json
    from pathlib import Path

    value = json.loads((Path(bundle) / op.metadata).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"metadata for {op.slot!r} must be a JSON object")
    return value


def cmd_verify(args: argparse.Namespace) -> int:
    from optima.registry import (
        Eligibility,
        KernelImpl,
        KernelRegistry,
        eligibility_from_metadata,
    )
    from optima.slots import SLOTS, get_slot, model_profile, slot_for_model
    from optima.verify import format_verify, verify_entry

    m = load_manifest(args.bundle)
    if not _recursive_scan_ok(args.bundle, manifest=m):  # vendored-tree guard (every .py, not just entries)
        return 2

    # Parse every known row once and run the complete bundle through the SAME
    # registration rules used by the live seam before loading any candidate source.
    # Per-row verification alone cannot detect two individually valid domains that
    # overlap and would make live routing ambiguous.
    metadata_by_row: dict[int, dict] = {}
    eligibility_by_row: dict[int, Eligibility] = {}
    domain_registry = KernelRegistry()

    def _domain_only_entry(*_args, **_kwargs):
        raise AssertionError("domain preflight entries are never invoked")

    for row_index, op in enumerate(m.ops):
        if op.slot not in SLOTS:
            continue
        label = f"{op.slot} variant={op.variant!r}"
        try:
            metadata = _declared_metadata(args.bundle, op)
            eligibility = eligibility_from_metadata(
                metadata, op.dtypes, op.architectures
            )
            domain_registry.register(
                KernelImpl(
                    slot=op.slot,
                    bundle_id=m.bundle_id,
                    entry=_domain_only_entry,
                    eligibility=eligibility,
                    variant=op.variant,
                )
            )
        except (OSError, ValueError) as exc:
            print(f"  [FAIL] {label}: invalid or ambiguous variant domain: {exc}")
            return 2
        metadata_by_row[row_index] = metadata
        eligibility_by_row[row_index] = eligibility

    import torch
    # Mirror the ACTUAL device resolution, including verify_collective's fallback:
    # a collective needs world_size GPUs, so a 1-GPU box silently runs gloo/CPU.
    ws = getattr(args, "world_size", None) or 2
    has_collective = any(op.slot in SLOTS and get_slot(op.slot).kind == "collective"
                         for op in m.ops)
    cuda_ok = torch.cuda.is_available() and (
        not has_collective or torch.cuda.device_count() >= ws)
    effective_device = args.device or ("cuda" if cuda_ok else "cpu")
    if effective_device == "cpu":
        print("[note] some or all of this verify runs on CPU: it checks op-correctness "
              "only — it does not predict GPU throughput, CUDA-graph capture, or the "
              "fidelity gates (see docs/GPU_SETUP.md).")
    rc = 0
    known_rows = 0
    context_inapplicable_rows = 0
    for row_index, op in enumerate(m.ops):
        label = f"{op.slot} variant={op.variant!r}"
        if op.slot not in SLOTS:
            print(f"  [SKIP] {label}: not a known slot on this validator")
            continue
        known_rows += 1
        metadata = metadata_by_row[row_index]
        model_key = args.model or metadata.get("model") or metadata.get("model_profile")
        if model_profile(model_key, op.slot) is not None:
            via = "via --model" if args.model else "declared in metadata"
            print(f"  [profile] {label}: model {model_key!r} ({via}) -> validator slot profile "
                  "(activation + low-bit metric)")
        slot = slot_for_model(op.slot, model_key)
        src = resolve_source(args.bundle, op)
        graph_safe = None if slot.kind == "op" else bool(
            metadata.get("graph_safe", False)
        )

        scan = scan_path(src)
        if not scan.ok:
            print(f"  [FAIL] {label}: failed policy scan")
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
                                       dtype_name=args.dtype,
                                       jitter_seed=args.seed,  # anti shape-branch, like per-op
                                       model_key=model_key,
                                       # rebuild plan (declared cuda_sources) must apply
                                       # in the ranks that load the kernel
                                       bundle_path=str(args.bundle),
                                       graph_safe=bool(graph_safe),
                                       eligibility=eligibility_by_row[row_index],
                                       tp_size=getattr(args, "tp_size", None),
                                       variant_name=op.variant)
            print(f"  [variant {op.variant!r}]")
            print(format_verify(result))
            if result.context_inapplicable:
                context_inapplicable_rows += 1
            elif not result.passed or (
                effective_device == "cuda" and not result.fully_verified
            ):
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
            graph_safe=graph_safe,
            eligibility_metadata=metadata,
            manifest_dtypes=op.dtypes,
            manifest_architectures=op.architectures,
            tp_size=getattr(args, "tp_size", None),
            world_size=getattr(args, "world_size", None),
            bundle_path=str(args.bundle),
            variant_name=op.variant,
        )
        print(f"  [variant {op.variant!r}]")
        print(format_verify(result))
        if result.context_inapplicable:
            context_inapplicable_rows += 1
        elif not result.passed:
            rc = 2
    if known_rows and context_inapplicable_rows == known_rows and rc == 0:
        print("no bundle variant is applicable to the selected verify context")
        rc = 2
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="optima",
        description=(
            "Optima validator harness.\n"
            "\n"
            "Commands by workflow:\n"
            "  develop a kernel (miner) ... slots, scan, verify\n"
            "  submit on-chain (miner) .... chain-register, chain-package,\n"
            "                               chain-submit, chain-status\n"
            "  referee + settlement ....... chain-validate, set-weights\n"
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

    sp = sub.add_parser(
        "model-provision",
        help=(
            "seal a clean exact model tree into a content-addressed external receipt "
            "(transient cache paths are rejected)"
        ),
    )
    sp.add_argument("model_root")
    sp.add_argument("publication_root")
    sp.add_argument("--expected-content-digest")
    sp.add_argument("--workers", type=int, default=4)
    sp.set_defaults(func=cmd_model_provision)

    sp = sub.add_parser(
        "release-verify",
        help="reopen and verify a signed chain-independent Optima Engine release",
    )
    sp.add_argument("release_root")
    sp.add_argument("--expected-public-key", required=True)
    sp.add_argument("--descriptor-digest")
    sp.set_defaults(func=cmd_release_verify)

    sp = sub.add_parser(
        "release-context",
        help="materialize a deterministic OCI build context from a verified release",
    )
    sp.add_argument("release_root")
    sp.add_argument("destination")
    sp.add_argument("--expected-public-key", required=True)
    sp.add_argument("--descriptor-digest")
    sp.set_defaults(func=cmd_release_context)

    sp = sub.add_parser(
        "set-weights",
        help="control-plane reconcile of the transactional global reward projection",
    )
    sp.add_argument("--intake-db", default="chain_intake/intake.sqlite3")
    sp.add_argument("--netuid", type=int, required=True)
    sp.add_argument("--network", default="finney",
                    help="named network or an explicit wss:// endpoint URL")
    sp.add_argument("--wallet", default="default")
    sp.add_argument("--hotkey", default="default")
    sp.add_argument("--half-life-blocks", type=int, required=True)
    sp.add_argument("--discovery-lifetime-blocks", type=int, required=True)
    sp.add_argument("--discovery-pool-ppm", type=int, required=True)
    sp.add_argument("--refresh-blocks", type=int, required=True)
    sp.add_argument(
        "--release-hold",
        default="",
        metavar="REASON",
        help="append an audited release of the current held publication; does not submit",
    )
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
                        help="finalized reveal -> private fetch -> immutable worker publication")
    sp.add_argument("--netuid", type=int, required=True)
    sp.add_argument("--network", required=True)
    sp.add_argument("--intake-only", action="store_true",
                    help="explicitly disable qualification, settlement, signing, and weights")
    sp.add_argument("--arena-id", default=None,
                    help="validator-owned registered arena selected from injected services")
    sp.add_argument("--intake-db", default="chain_intake/intake.sqlite3")
    sp.add_argument("--private-root", default="chain_intake/private",
                    help="validator-private 0700/0600 fetch storage")
    sp.add_argument("--publication-root", default="chain_intake/worker",
                    help="immutable 0555/0444 worker-readable publication storage")
    sp.add_argument("--interval", type=float, default=60.0, help="seconds between passes")
    sp.add_argument("--once", action="store_true", help="single pass, then exit")
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
    sp.add_argument("--tp-size", type=int, default=None, dest="tp_size",
                    help="tensor-parallel size for capability-aware non-collective verify")
    sp.add_argument("--model", default=None,
                    help="validator model key for the per-model slot profile (activation + "
                         "low-bit metric), e.g. MiniMax-M3. Default: the model declared in the "
                         "op's metadata (dev convenience); production uses the served-model key.")
    sp.set_defaults(func=cmd_verify)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
