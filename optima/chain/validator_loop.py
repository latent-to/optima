"""The validator loop: chain commitments → fetch → evaluate → settle → weights.

One ``run_pass`` is the whole referee cycle against live chain state; ``run_validator``
repeats it forever with per-pass fault isolation (one bad submission or one RPC
hiccup must never kill the loop — reject, record, continue).

Trust boundaries, in order:
- the PAYLOAD is untrusted (fail-quiet decode, ``optima.chain.payload``);
- the ARTIFACT is untrusted (size-capped fetch, hostile-archive extraction, and the
  extracted tree must re-hash to the committed content hash, ``optima.chain.fetch``);
- the BUNDLE is untrusted (evaluated out-of-process via ``python -m optima.cli`` —
  this module never imports miner code, same discipline as ``cmd_verify``);
- weight POLICY is not this module's business: it consumes
  ``PerSlotSettleResult.weights`` from the Ledger so the emission scheme (currently
  per-slot king-of-the-hill; NOT winner-take-all-forever) swaps without touching
  chain I/O.

Every processed submission is recorded in the Ledger (scores for settlement,
EvalRecords for the audit trail + retry suppression), so restarts re-derive state
from the ledger file instead of replaying work — the "re-derive, don't replay"
pattern from SUBNET_BLUEPRINT §2.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from optima import chain
from optima.chain.fetch import FetchError, fetch_bundle
from optima.chain.payload import SubmissionRef, decode_payload

logger = logging.getLogger("optima.chain.validator")

# A "round" is one settlement window; by default it advances with the subnet tempo.
DEFAULT_ROUND_BLOCKS = 360
# Re-assert weights at least this often even when unchanged (activity cutoff prunes
# validators that go quiet; the subnet's cutoff is thousands of blocks — one tempo
# of headroom is comfortable).
DEFAULT_WEIGHTS_REFRESH_BLOCKS = 360

EVAL_TIMEOUT_S = 3600.0


@dataclass
class EvalOutcome:
    """What an evaluator says about one fetched bundle."""
    passed: bool
    score: float
    kl_mean: float = 0.0
    slot: str = ""
    detail: str = ""


# An Evaluator takes the fetched bundle directory and returns an EvalOutcome.
# It must never raise for a *bad bundle* — that's a failed outcome; raising is
# reserved for validator-side faults (which fail the pass, not the submission).
Evaluator = Callable[[Path], EvalOutcome]


@dataclass
class PassResult:
    block: int
    round_id: int
    seen: int = 0
    new: list[str] = field(default_factory=list)        # content hashes processed
    rejected: dict[str, str] = field(default_factory=dict)  # hash/hotkey -> reason
    copies: list[str] = field(default_factory=list)
    evaluated: dict[str, bool] = field(default_factory=dict)  # hash -> passed
    weights: dict[str, float] = field(default_factory=dict)
    weights_pushed: bool = False


# --------------------------------------------------------------------------- #
# Evaluators — all out-of-process
# --------------------------------------------------------------------------- #

def verify_evaluator(device: str = "cpu", dtype: str = "float32",
                     timeout_s: float = EVAL_TIMEOUT_S) -> Evaluator:
    """PLUMBING-ONLY evaluator: runs ``optima verify`` and scores pass/fail as
    1.0/0.0. It proves the loop end-to-end (fetch → gates → settle → weights)
    without a GPU; the "score" is NOT a throughput measurement and must never be
    used for real emissions — production wires ``command_evaluator`` to the full
    ``optima evaluate`` gate chain on the GPU box."""
    def _run(bundle_dir: Path) -> EvalOutcome:
        cmd = [sys.executable, "-m", "optima.cli", "verify", str(bundle_dir),
               "--device", device, "--dtype", dtype]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return EvalOutcome(False, 0.0, detail=f"verify timed out after {timeout_s}s")
        tail = (proc.stdout + proc.stderr)[-2000:]
        return EvalOutcome(proc.returncode == 0, 1.0 if proc.returncode == 0 else 0.0,
                           detail=tail)
    return _run


def command_evaluator(template: str, timeout_s: float = EVAL_TIMEOUT_S) -> Evaluator:
    """Run an arbitrary eval command per bundle: ``template`` is shell text with
    ``{bundle}`` and ``{report}`` placeholders. Contract: exit 0 = passed the gate
    chain; the command SHOULD write JSON ``{"score":float,"kl_mean":float,
    "slot":str}`` to ``{report}`` (missing report on success = score 1.0). This is
    how the GPU box runs the real ``optima evaluate`` under this loop."""
    def _run(bundle_dir: Path) -> EvalOutcome:
        report = bundle_dir.parent / f".{bundle_dir.name}.report.json"
        cmd = template.format(bundle=str(bundle_dir), report=str(report))
        try:
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                  timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return EvalOutcome(False, 0.0, detail=f"eval timed out after {timeout_s}s")
        passed = proc.returncode == 0
        score, kl, slot = (1.0 if passed else 0.0), 0.0, ""
        if report.exists():
            try:
                data = json.loads(report.read_text())
                score = float(data.get("score", score))
                kl = float(data.get("kl_mean", 0.0))
                slot = str(data.get("slot", ""))
            except (json.JSONDecodeError, TypeError, ValueError):
                passed, score = False, 0.0
        return EvalOutcome(passed, score if passed else 0.0, kl_mean=kl, slot=slot,
                           detail=(proc.stdout + proc.stderr)[-2000:])
    return _run


# --------------------------------------------------------------------------- #
# One referee pass
# --------------------------------------------------------------------------- #

def _bundle_slot(bundle_dir: Path) -> str:
    """First declared slot — mirrors cmd_evaluate's ledger recording. Manifest
    parsing is the validator's own trusted code (static TOML; imports nothing)."""
    from optima.manifest import load_manifest

    try:
        m = load_manifest(bundle_dir)
        return m.ops[0].slot if m.ops else ""
    except Exception as e:  # noqa: BLE001 — a bad manifest is a bad submission
        logger.warning("manifest unreadable in fetched bundle %s: %s", bundle_dir, e)
        return ""


def _load_weights_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def run_pass(subtensor, wallet, netuid: int, *, ledger_path: str, bundles_dir: str,
             evaluator: Evaluator, margin: float = 0.02,
             round_blocks: int = DEFAULT_ROUND_BLOCKS,
             weights_refresh_blocks: int = DEFAULT_WEIGHTS_REFRESH_BLOCKS,
             dry_run_weights: bool = False) -> PassResult:
    """One full referee cycle. Per-submission failures are recorded and contained;
    an exception out of this function is a validator-side fault (RPC, disk)."""
    from optima.commit_reveal import EvalRecord, Ledger, RevealError, make_commitment
    from optima.compat import PINNED_SGLANG
    from optima.copy_fingerprint import (
        bundle_fingerprint,
        bundle_slot_file_fingerprints,
        bundle_slot_fingerprints,
        bundle_structural_fingerprint,
    )

    block = int(subtensor.get_current_block())
    round_id = block // round_blocks
    res = PassResult(block=block, round_id=round_id)

    revealed = chain.read_revealed_commitments(subtensor, netuid)
    refs: list[SubmissionRef] = []
    for hotkey, rc in revealed.items():
        ref = decode_payload(hotkey, rc.block, rc.data)
        if ref is not None:
            refs.append(ref)
    # Chain order = anti-copy priority: replay into the ledger sorted by reveal block.
    refs.sort(key=lambda r: (r.block, r.hotkey))
    res.seen = len(refs)

    led = Ledger.load(ledger_path)
    known_reveals = {(r.hotkey, r.content_hash) for r in led.reveals}

    for ref in refs:
        key = (ref.hotkey, ref.content_hash)
        if key in known_reveals or led.is_known(ref.hotkey, ref.content_hash):
            continue  # already processed (or already rejected) — re-derive, don't replay
        res.new.append(ref.content_hash)

        try:
            bundle_dir = fetch_bundle(ref.url, ref.content_hash, bundles_dir)
        except FetchError as e:
            logger.warning("submission %s… by %s rejected: %s",
                           ref.content_hash[:16], ref.hotkey, e)
            res.rejected[ref.content_hash] = str(e)
            led.record_eval(EvalRecord(ref.hotkey, ref.content_hash, slot="",
                                       round_id=round_id, score=0.0, passed=False,
                                       dq_reason=f"fetch: {e}"))
            continue

        slot = _bundle_slot(bundle_dir)
        try:
            fingerprints = dict(
                fingerprint=bundle_fingerprint(bundle_dir),
                structural_fingerprint=bundle_structural_fingerprint(bundle_dir),
                slot_fingerprints=bundle_slot_fingerprints(bundle_dir),
                slot_file_fingerprints=bundle_slot_file_fingerprints(bundle_dir),
            )
        except Exception as e:  # noqa: BLE001 — malformed manifest/sources = bad submission
            logger.warning("submission %s… by %s rejected: unfingerprintable: %s",
                           ref.content_hash[:16], ref.hotkey, e)
            res.rejected[ref.content_hash] = f"unfingerprintable: {e}"
            led.record_eval(EvalRecord(ref.hotkey, ref.content_hash, slot=slot,
                                       round_id=round_id, score=0.0, passed=False,
                                       dq_reason=f"unfingerprintable: {e}"))
            continue
        salt = f"chain:{ref.block}"
        led.commit(ref.hotkey, make_commitment(ref.content_hash, ref.hotkey, salt), round_id)
        try:
            rev = led.reveal(ref.hotkey, ref.content_hash, salt, round_id, **fingerprints)
        except RevealError as e:  # cannot happen for a commit we just made; belt+braces
            res.rejected[ref.content_hash] = str(e)
            continue
        if not rev.original:
            logger.info("submission %s… by %s is a COPY of an earlier commit; skipping eval",
                        ref.content_hash[:16], ref.hotkey)
            res.copies.append(ref.content_hash)
            led.record_eval(EvalRecord(ref.hotkey, ref.content_hash, slot=slot,
                                       round_id=round_id, score=0.0, passed=False,
                                       dq_reason="copy"))
            continue

        outcome = evaluator(bundle_dir)
        slot = outcome.slot or slot
        led.record_score(ref.hotkey, ref.content_hash, round_id, outcome.score,
                         outcome.kl_mean, outcome.passed,
                         sglang_version=PINNED_SGLANG, slot=slot)
        led.record_eval(EvalRecord(ref.hotkey, ref.content_hash, slot=slot,
                                   round_id=round_id, score=outcome.score,
                                   passed=outcome.passed, mean_kl=outcome.kl_mean,
                                   dq_reason="" if outcome.passed else "failed gates"))
        res.evaluated[ref.content_hash] = outcome.passed
        logger.info("evaluated %s… by %s: passed=%s score=%.4f slot=%s",
                    ref.content_hash[:16], ref.hotkey, outcome.passed, outcome.score, slot)

    settle = led.settle_per_slot(round_id, margin=margin,
                                 current_sglang_version=PINNED_SGLANG)
    led.save(ledger_path)
    res.weights = dict(settle.weights)

    # Push weights when they changed, or on the refresh cadence (stay "active").
    state_path = Path(str(ledger_path) + ".weights_state.json")
    state = _load_weights_state(state_path)
    changed = res.weights != state.get("weights")
    stale = block - int(state.get("block", 0)) >= weights_refresh_blocks
    if res.weights and (changed or stale):
        pushed = chain.set_weights(subtensor, wallet, netuid, res.weights,
                                   dry_run=dry_run_weights)
        res.weights_pushed = bool(pushed.get("submitted"))
        if res.weights_pushed:
            state_path.write_text(json.dumps({"block": block, "weights": res.weights}))
    return res


def run_validator(subtensor, wallet, netuid: int, *, ledger_path: str, bundles_dir: str,
                  evaluator: Evaluator, margin: float = 0.02, interval_s: float = 60.0,
                  once: bool = False, dry_run_weights: bool = False,
                  round_blocks: int = DEFAULT_ROUND_BLOCKS,
                  max_consecutive_failures: int = 10) -> Optional[PassResult]:
    """The daemon: run passes forever (or ``once``). A failing pass is logged and
    retried with linear backoff; ``max_consecutive_failures`` in a row exits nonzero
    so a supervisor restarts us with fresh connections (crash-only discipline)."""
    failures = 0
    last: Optional[PassResult] = None
    while True:
        try:
            last = run_pass(subtensor, wallet, netuid, ledger_path=ledger_path,
                            bundles_dir=bundles_dir, evaluator=evaluator, margin=margin,
                            round_blocks=round_blocks, dry_run_weights=dry_run_weights)
            failures = 0
            logger.info("pass @block %d: seen=%d new=%d copies=%d rejected=%d weights=%s%s",
                        last.block, last.seen, len(last.new), len(last.copies),
                        len(last.rejected), last.weights,
                        " (pushed)" if last.weights_pushed else "")
        except Exception:  # noqa: BLE001 — validator-side fault; contain and retry
            failures += 1
            logger.exception("validator pass failed (%d consecutive)", failures)
            if failures >= max_consecutive_failures:
                raise
        if once:
            return last
        time.sleep(interval_s * (1 + min(failures, 5)))
