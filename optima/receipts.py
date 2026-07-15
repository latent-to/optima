"""Seam execution receipts — positive accounting evidence for the referee.

The failure mode this closes (hit for real on 2026-07-07): the candidate engine
comes up WITHOUT the seam (missing ``optima.pth``, bad env, bundle load failure
falling back to baseline) and the eval happily scores stock-vs-stock — identical
logits, KL exactly 0.0, accuracy delta 0.0, verdict PASS. ``seam.activate()``
deliberately never wedges the engine on a bad bundle, so the *engine* can't be
the one to fail; the *eval driver* must demand positive evidence.

Evidence lives where the seam lives — in sglang's spawned scheduler ranks — so it
travels by file: the driver sets ``OPTIMA_SEAM_RECEIPT_DIR`` for the candidate
launch, ranks write receipts there, the driver requires them:

  * ``active``      — bundle loaded + registry enabled in a rank (seam.activate).
  * ``load_failed`` — a rank ATTEMPTED the bundle load and fell back to baseline;
                      lets the driver report "bad bundle" instead of "no bootstrap".
  * ``fired``       — the registry SELECTED the miner impl for a slot at least once;
                      this is routing evidence only.
  * ``completed``   — a dispatcher successfully produced the model-facing output
                      after invoking the selected implementation; once/slot/process.
  * ``fallback``    — a selected path failed and the dispatcher served the trusted
                      baseline instead; once/slot/process and disqualifying.

``completed`` is stronger than ``fired`` but remains diagnostic execution evidence,
not hostile-code proof. Candidate Python shares the scheduler process today and can
forge process-local state; complete-engine isolation plus external qualification is
the crown boundary.

No env var set -> every helper is a silent no-op (verify paths, unit tests, and
baseline launches don't produce receipt litter).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

logger = logging.getLogger("optima.receipts")

_SAFE_RE = re.compile(r"[^0-9A-Za-z._\-]+")
_IDENTITY_KINDS = frozenset(
    {
        "active",
        "load_failed",
        "fired",
        "completed",
        "fallback",
        "audit",
        "aot_loaded",
        "aot_invoked",
    }
)
# Include the receipt directory so one long-lived process can participate in
# independent launches without an earlier launch suppressing the later receipt.
_ONCE: set[tuple[str, int, str, str]] = set()
_ONCE_LOCK = threading.Lock()


def _dir() -> str:
    return os.environ.get("OPTIMA_SEAM_RECEIPT_DIR", "").strip()


def _resolved_dir(raw: str) -> Path:
    try:
        return Path(raw).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        return Path(os.path.abspath(os.path.expanduser(raw)))


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or not raw.isascii() or not raw.isdecimal():
        return None
    return int(raw)


def identity() -> dict:
    """Best-effort scheduler-member identity, always including a stable PID."""
    pid = os.getpid()
    rank: Optional[int] = None
    world_size: Optional[int] = None
    try:
        import torch.distributed as dist  # deferred: CPU tooling need not initialize it

        if dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())
    except Exception:  # noqa: BLE001 - receipts must never break model execution
        pass
    if rank is None or world_size is None:
        rank = _env_int("RANK")
        world_size = _env_int("WORLD_SIZE")
    if (
        rank is None
        or world_size is None
        or world_size < 1
        or rank < 0
        or rank >= world_size
    ):
        rank = world_size = -1
    return {"pid": pid, "rank": rank, "world_size": world_size}


def _write_to(root: Path, kind: str, payload: dict, *, tag: str = "") -> bool:
    try:
        body = dict(payload)
        if kind in _IDENTITY_KINDS:
            # Detected identity is authoritative over caller-supplied fields.
            body = {**body, **identity()}
        root.mkdir(parents=True, exist_ok=True)
        suffix = f".{_SAFE_RE.sub('_', tag)}" if tag else ""
        p = root / f"{kind}{suffix}.{os.getpid()}.json"
        p.write_text(json.dumps(body, sort_keys=True))
        return True
    except Exception:  # noqa: BLE001
        logger.exception("optima: receipt write failed (kind=%s)", kind)
        return False


def write(kind: str, payload: dict, *, tag: str = "") -> None:
    """Write one receipt file; never raises (a receipt must not break an engine)."""
    raw = _dir()
    if raw:
        _write_to(_resolved_dir(raw), kind, payload, tag=tag)


def _write_execution_once(
    kind: str, slot: str, *, error: BaseException | None = None
) -> None:
    """Write one slot execution receipt without adding hot-path file churn."""
    rdir = _dir()
    if not rdir:
        # Do not consume the guard: a later independently receipted launch in this
        # process must still produce evidence.
        return
    root = _resolved_dir(rdir)
    key = (str(root), os.getpid(), kind, slot)
    with _ONCE_LOCK:
        if key in _ONCE:
            return
        payload = {"slot": slot}
        if error is not None:
            try:
                message = str(error)[:512]
            except Exception:  # noqa: BLE001 - hostile exception formatting is diagnostic
                message = "<unprintable exception>"
            payload.update(error_type=type(error).__name__, error=message)
        if _write_to(root, kind, payload, tag=slot):
            _ONCE.add(key)


def completed(slot: str) -> None:
    """Record successful candidate output production once for this slot/process."""
    _write_execution_once("completed", slot)


def fallback(slot: str, error: BaseException) -> None:
    """Record that a selected path failed and trusted stock was served."""
    _write_execution_once("fallback", slot, error=error)


class ReceiptFormatError(RuntimeError):
    """A receipt file exists but is unreadable or not a JSON object."""


def collect(rdir: str | Path, kind: str) -> list[dict]:
    """Strictly read all receipts of ``kind``; malformed evidence fails closed."""
    out: list[dict] = []
    root = Path(rdir)
    if not root.is_dir():
        return out
    for p in sorted(root.glob(f"{kind}*.json")):
        try:
            payload = json.loads(p.read_text())
        except (OSError, ValueError) as exc:  # noqa: PERF203
            raise ReceiptFormatError(f"invalid receipt {p}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ReceiptFormatError(f"invalid receipt {p}: expected a JSON object")
        out.append(payload)
    return out


def require(rdir: str | Path, kind: str, *, context: str) -> list[dict]:
    """Return receipts of ``kind`` or raise with a diagnosis — the eval-side gate."""
    got = collect(rdir, kind)
    if got:
        return got
    failed = collect(rdir, "load_failed")
    if failed:
        raise RuntimeError(
            f"{context}: seam rank(s) attempted the bundle load and FELL BACK to baseline "
            f"(load_failed receipts: {failed}). The run would have scored stock-vs-stock; "
            "fix the bundle, do not trust any output from this launch."
        )
    raise RuntimeError(
        f"{context}: no '{kind}' seam receipt was written by any engine rank. The candidate "
        "ran WITHOUT the miner kernel (stock-vs-stock) — likely missing optima.pth bootstrap "
        "in the engine interpreter, OPTIMA env not reaching spawned ranks, or the seamed "
        "module was never imported by this engine config. Refusing to score a phantom."
    )


def _exact_int(value, *, minimum: int = 0) -> Optional[int]:
    if type(value) is not int or value < minimum:  # bool is intentionally invalid
        return None
    return value


def _validated_identity(receipt: dict) -> tuple[int, int, int] | None:
    pid = _exact_int(receipt.get("pid"), minimum=1)
    rank = _exact_int(receipt.get("rank"), minimum=-1)
    world_size = _exact_int(receipt.get("world_size"), minimum=-1)
    if pid is None or rank is None or world_size is None:
        return None
    if (rank, world_size) == (-1, -1):
        return pid, rank, world_size
    if world_size < 1 or rank < 0 or rank >= world_size:
        return None
    return pid, rank, world_size


def _expected_members(
    observed: list[dict],
    members: list[dict],
    *,
    expected_member_count: int | None,
) -> tuple[
    str,
    list[str],
    list[dict],
    list[dict],
    bool,
    dict[int, tuple[int, int]],
]:
    """Resolve members without allowing observed completions to hide a silent rank."""
    malformed: list[dict] = []
    duplicates: list[dict] = []
    if members:
        seen: set[int] = set()
        member_identities: dict[int, tuple[int, int]] = {}
        for receipt in members:
            ident = _validated_identity(receipt)
            if ident is None:
                malformed.append(receipt)
                continue
            pid = ident[0]
            if pid in seen:
                duplicates.append(receipt)
            seen.add(pid)
            member_identities[pid] = (ident[1], ident[2])
        labels = [f"pid:{pid}" for pid in sorted(seen)]
        count_ok = expected_member_count is None or len(labels) == expected_member_count
        known = [identity for identity in member_identities.values() if identity != (-1, -1)]
        if known and len(known) != len(member_identities):
            malformed.extend(members)
        elif known:
            world_sizes = {world_size for _rank, world_size in known}
            ranks = [rank for rank, _world_size in known]
            if len(world_sizes) != 1:
                malformed.extend(members)
            else:
                world_size = next(iter(world_sizes))
                if (
                    world_size != len(member_identities)
                    or (expected_member_count is not None
                        and world_size != expected_member_count)
                    or set(ranks) != set(range(world_size))
                    or len(set(ranks)) != len(ranks)
                ):
                    malformed.extend(members)
                    count_ok = False
        return (
            "pid",
            labels,
            malformed,
            duplicates,
            count_ok,
            member_identities,
        )

    identities = [_validated_identity(receipt) for receipt in observed]
    if not identities or any(ident is None for ident in identities):
        malformed.extend(
            receipt
            for receipt, ident in zip(observed, identities)
            if ident is None
        )
        return "unproven", [], malformed, duplicates, False, {}
    known = [ident for ident in identities if ident is not None]
    world_sizes = {ident[2] for ident in known}
    if -1 in world_sizes or len(world_sizes) != 1:
        malformed.extend(observed)
        return "unproven", [], malformed, duplicates, False, {}
    world_size = next(iter(world_sizes))
    assert world_size > 0
    labels = [f"rank:{rank}" for rank in range(world_size)]
    count_ok = expected_member_count is None or world_size == expected_member_count
    return "rank", labels, malformed, duplicates, count_ok, {}


def coverage_matrix(
    observed: Iterable[dict],
    *,
    expected_slots: Iterable[str],
    member_receipts: Iterable[dict] = (),
    expected_member_count: int | None = None,
    count_field: str | None = None,
    min_count: int = 1,
) -> dict:
    """Build fail-closed per-slot/per-member diagnostic coverage."""
    got = list(observed)
    members = list(member_receipts)
    raw_slots = list(expected_slots)
    if any(not isinstance(slot, str) or not slot for slot in raw_slots):
        raise ValueError("expected_slots must contain non-empty strings")
    slots = sorted(set(raw_slots))
    if expected_member_count is not None and (
        type(expected_member_count) is not int or expected_member_count < 1
    ):
        raise ValueError("expected_member_count must be a positive integer or None")
    if type(min_count) is not int or min_count < 1:
        raise ValueError("min_count must be a positive integer")
    (
        basis,
        expected_members,
        malformed,
        duplicates,
        member_count_ok,
        active_identities,
    ) = (
        _expected_members(
            got, members, expected_member_count=expected_member_count
        )
    )
    expected_pairs = {
        (slot, member) for slot in slots for member in expected_members
    }
    counts: dict[tuple[str, str], int] = {}
    unexpected: list[dict] = []
    for receipt in got:
        slot = receipt.get("slot")
        ident = _validated_identity(receipt)
        if not isinstance(slot, str) or not slot or ident is None:
            malformed.append(receipt)
            continue
        if basis == "pid":
            member = f"pid:{ident[0]}"
            active_identity = active_identities.get(ident[0])
            if (
                active_identity is not None
                and active_identity != (-1, -1)
                and active_identity != (ident[1], ident[2])
            ):
                malformed.append(receipt)
                continue
        elif basis == "rank" and ident[1] >= 0:
            member = f"rank:{ident[1]}"
        else:
            malformed.append(receipt)
            continue
        if slot not in slots or member not in expected_members:
            unexpected.append(receipt)
            continue
        if count_field is None:
            count = 1
        else:
            count = _exact_int(receipt.get(count_field))
            if count is None:
                malformed.append(receipt)
                continue
        key = (slot, member)
        counts[key] = counts.get(key, 0) + count
        if count_field is None and counts[key] > 1:
            duplicates.append(receipt)

    if (
        basis == "pid"
        and active_identities
        and all(identity == (-1, -1) for identity in active_identities.values())
    ):
        # Bootstrap may precede process-group initialization, so early active
        # receipts can be PID-only. For multi-member execution, the later
        # completions must then prove one coherent distributed identity per PID.
        target_count = expected_member_count or len(active_identities)
        derived: dict[int, set[tuple[int, int]]] = {
            pid: set() for pid in active_identities
        }
        for receipt in got:
            ident = _validated_identity(receipt)
            if ident is not None and ident[0] in derived:
                derived[ident[0]].add((ident[1], ident[2]))
        if target_count > 1:
            identities = [
                next(iter(values))
                for values in derived.values()
                if len(values) == 1 and (-1, -1) not in values
            ]
            if (
                len(identities) != len(derived)
                or {world for _rank, world in identities} != {target_count}
                or {rank for rank, _world in identities} != set(range(target_count))
            ):
                malformed.extend(got)
        else:
            known = {
                identity
                for values in derived.values()
                for identity in values
                if identity != (-1, -1)
            }
            if known and known != {(0, 1)}:
                malformed.extend(got)
    present = {pair for pair, count in counts.items() if count >= min_count}
    missing = sorted(expected_pairs - present)
    short = sorted(
        (slot, member, counts.get((slot, member), 0))
        for slot, member in expected_pairs
        if 0 < counts.get((slot, member), 0) < min_count
    )
    return {
        "ok": (
            bool(slots)
            and bool(expected_members)
            and member_count_ok
            and not missing
            and not malformed
            and not unexpected
            and not duplicates
        ),
        "basis": basis,
        "expected_slots": slots,
        "members": expected_members,
        "expected_member_count": expected_member_count,
        "observed_member_count": len(expected_members),
        "member_count_ok": member_count_ok,
        "expected_pairs": len(expected_pairs),
        "covered_pairs": len(expected_pairs & present),
        "missing": [
            {"slot": slot, "member": member} for slot, member in missing
        ],
        "short": [
            {"slot": slot, "member": member, "count": count, "required": min_count}
            for slot, member, count in short
        ],
        "malformed": malformed,
        "unexpected": unexpected,
        "duplicates": duplicates,
    }


def completed_gate(
    completed_receipts: Iterable[dict],
    *,
    expected_slots: Iterable[str],
    member_receipts: Iterable[dict] = (),
    expected_member_count: int | None = None,
    fallback_receipts: Iterable[dict] = (),
) -> tuple[bool, str]:
    """Require one completion per expected slot/member and zero selected fallbacks."""
    complete = list(completed_receipts)
    members = list(member_receipts)
    fallbacks = list(fallback_receipts)
    detail = coverage_matrix(
        complete,
        expected_slots=expected_slots,
        member_receipts=members,
        expected_member_count=expected_member_count,
    )
    # This directory is fresh per launch. Any fallback file—including a stale,
    # unexpected or semantically malformed one—makes the evidence incoherent.
    ok = detail["ok"] and not fallbacks
    desc = (
        f"completed coverage {detail['covered_pairs']}/{detail['expected_pairs']} "
        f"slot/member pairs (basis={detail['basis']})"
    )
    if detail["missing"]:
        desc += f"; missing={detail['missing']}"
    if detail["malformed"]:
        desc += f"; malformed={len(detail['malformed'])}"
    if detail["unexpected"]:
        desc += f"; unexpected={len(detail['unexpected'])}"
    if detail["duplicates"]:
        desc += f"; duplicates={len(detail['duplicates'])}"
    if not detail["member_count_ok"]:
        desc += (
            f"; members={detail['observed_member_count']}"
            f"/{detail['expected_member_count']}"
        )
    if fallbacks:
        desc += f"; selected-path fallbacks={fallbacks}"
    return ok, desc
