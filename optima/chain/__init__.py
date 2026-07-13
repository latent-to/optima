"""On-chain I/O: finalized submissions, global weights, and preflight checks.

Architecture mirrors the sglang seam. **Pure helpers** (weight-vector math, hotkey↔uid
mapping) carry no SDK and no network, so they are unit-tested directly. **Thin RPC
wrappers** lazily ``import bittensor`` and are the only code that touches the chain —
so the package still imports, and the test suite still runs, with no SDK installed.

The exact SDK methods called here are pinned by ``optima chain-compat``
(optima/chain_canary.py); run it after any bittensor bump, and the wrappers' calls
stay a thin, auditable layer over what that canary asserts.

Submissions ride the chain's NATIVE commit-reveal (SUBNET_BLUEPRINT §3): a miner
posts a timelock-encrypted payload (``set_reveal_commitment``, ≤1024 bytes,
drand-encrypted until the reveal round — nobody can read the bundle URL before
reveal, and the reveal block is the anti-copy priority timestamp). The validator
reads finalized storage and events in canonical order before transactional intake.
The older salted-hash transport (``set_commitment``/``get_all_commitments``)
remains for compatibility. Settlement and weight-publication authority live in
their dedicated control-plane modules, not in these thin SDK wrappers.
"""

from __future__ import annotations

import logging
import math
import operator
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("optima.chain")

# Yuma-consensus version stamped on set_weights. A coordinated subnet parameter,
# bumped deliberately (like PINNED_SGLANG) so every validator agrees.
WEIGHTS_VERSION_KEY = 1
CHAIN_REVEAL_HISTORY_CAP = 10
MAX_REVEAL_HISTORY_PAGES = 4_096
MAX_REVEAL_HISTORY_ROWS = 1_000_000
MAX_REVEAL_EVENT_BLOCKS = 4_096
_BLOCK_HASH_RE = re.compile(r"0x[0-9a-fA-F]{64}\Z")


class ChainRevealHistoryError(RuntimeError):
    """Finalized storage and event history cannot prove exact reveal order."""

    validator_fault = True
    retryable = False


class ChainWeightStateError(RuntimeError):
    """The validator's active on-chain vector cannot be read or projected safely."""

    validator_fault = True
    retryable = False


@dataclass
class Commitment:
    """A hotkey's current on-chain commitment — for Optima, the salted commit hash."""
    hotkey: str
    data: str
    block: int = 0


@dataclass(frozen=True)
class RevealedCommitment:
    """One reveal bound to its canonical finalized event position."""

    hotkey: str
    data: str
    block: int
    block_hash: str
    event_index: int
    extrinsic_index: int | None = None

    @property
    def priority_key(self) -> tuple[int, int, str, str]:
        return self.block, self.event_index, self.hotkey, self.data


@dataclass(frozen=True)
class FinalizedRevealSnapshot:
    finalized_block: int
    finalized_block_hash: str
    reveals: tuple[RevealedCommitment, ...]

    def __post_init__(self) -> None:
        if (
            type(self.finalized_block) is not int
            or self.finalized_block < 0
            or _BLOCK_HASH_RE.fullmatch(self.finalized_block_hash) is None
            or type(self.reveals) is not tuple
            or any(type(row) is not RevealedCommitment for row in self.reveals)
            or tuple(row.priority_key for row in self.reveals)
            != tuple(sorted(row.priority_key for row in self.reveals))
            or any(row.block > self.finalized_block for row in self.reveals)
        ):
            raise ChainRevealHistoryError("finalized reveal snapshot is malformed")


@dataclass(frozen=True)
class _RevealEventOccurrence:
    hotkey: str
    block: int
    block_hash: str
    event_index: int
    extrinsic_index: int | None


@dataclass
class MetagraphView:
    """A minimal, SDK-free snapshot of the metagraph used for weight-setting + seeding."""
    netuid: int
    block: int
    block_hash: str
    uids: list[int] = field(default_factory=list)
    hotkeys: list[str] = field(default_factory=list)  # index-aligned with uids
    validator_permit: list[bool] = field(default_factory=list)
    last_update: list[int] = field(default_factory=list)

    def uid_of(self, hotkey: str) -> Optional[int]:
        try:
            return self.uids[self.hotkeys.index(hotkey)]
        except ValueError:
            return None


@dataclass(frozen=True)
class ValidatorWeightSnapshot:
    """One validator's authoritative sparse vector and last-update block."""

    weights: dict[str, float]
    last_update_block: int


# --------------------------------------------------------------------------- #
# Pure helpers — exercised with no chain
# --------------------------------------------------------------------------- #

def normalize(weights: dict[str, float]) -> dict[str, float]:
    """Scale to sum 1.0, dropping non-positive entries. Empty / all-zero -> {}."""
    pos = {k: float(v) for k, v in weights.items() if v and v > 0}
    total = sum(pos.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in pos.items()}


def weights_to_uid_vector(weights_by_hotkey: dict[str, float],
                          metagraph: MetagraphView) -> tuple[list[int], list[float]]:
    """Map ``{hotkey: weight}`` to ``(uids, weights)`` for set_weights, aligned to the
    *live* metagraph. A missing positive recipient is an authority fault: silently
    redistributing its share would change the validator's settled policy."""
    positive: dict[str, float] = {}
    for hotkey, raw_weight in weights_by_hotkey.items():
        if (
            not isinstance(hotkey, str)
            or not hotkey
            or isinstance(raw_weight, bool)
            or not isinstance(raw_weight, (int, float))
        ):
            raise ChainWeightStateError("weight projection contains an invalid entry")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0:
            raise ChainWeightStateError(
                f"weight projection contains an invalid value for {hotkey!r}"
            )
        if weight > 0:
            positive[hotkey] = weight
    missing = sorted(
        hotkey for hotkey in positive if metagraph.uid_of(hotkey) is None
    )
    if missing:
        raise ChainWeightStateError(
            "positive-weight hotkeys are absent from the live metagraph: "
            + ", ".join(missing[:16])
        )
    norm = normalize(positive)
    uids = [metagraph.uid_of(hk) for hk in norm]
    if (
        any(type(uid) is not int or uid < 0 for uid in uids)
        or len(set(uids)) != len(uids)
    ):
        raise ChainWeightStateError(
            "live metagraph maps weight recipients to invalid or duplicate UIDs"
        )
    weights = [norm[hk] for hk in norm]
    return uids, weights


def _weight_uint(raw: object, field: str, *, maximum: int | None = None) -> int:
    if isinstance(raw, bool):
        raise ChainWeightStateError(f"invalid {field}")
    try:
        value = operator.index(raw)
    except (TypeError, OverflowError):
        raise ChainWeightStateError(f"invalid {field}") from None
    if value < 0 or (maximum is not None and value > maximum):
        raise ChainWeightStateError(f"invalid {field}")
    return value


def read_validator_weight_snapshot(
    subtensor,
    netuid: int,
    validator_hotkey: str,
    *,
    metagraph_view: MetagraphView | None = None,
) -> ValidatorWeightSnapshot:
    """Read one validator's live sparse row without trusting a local journal.

    Bittensor 10's default lite metagraph may omit the dense ``W`` matrix. The
    authoritative API is ``subtensor.weights()``, paired with metagraph UIDs and
    ``last_update`` for reconciliation after process restarts.
    """

    if type(netuid) is not int or netuid < 0:
        raise ChainWeightStateError("netuid must be a non-negative integer")
    if not isinstance(validator_hotkey, str) or not validator_hotkey:
        raise ChainWeightStateError("validator hotkey must be non-empty")
    if metagraph_view is not None:
        if type(metagraph_view) is not MetagraphView or metagraph_view.netuid != netuid:
            raise ChainWeightStateError("supplied metagraph view is not authoritative")
        hotkeys = list(metagraph_view.hotkeys)
        raw_uids = list(metagraph_view.uids)
        raw_last_updates = list(metagraph_view.last_update)
    else:
        try:
            metagraph = subtensor.metagraph(netuid=netuid)
            hotkeys = list(metagraph.hotkeys)
            raw_uids = list(metagraph.uids)
            raw_last_updates = list(metagraph.last_update)
        except Exception as exc:
            raise ChainWeightStateError(
                f"cannot fetch metagraph for active-weight verification: {exc}"
            ) from None
    if len(raw_uids) != len(hotkeys) or len(raw_last_updates) != len(hotkeys):
        raise ChainWeightStateError("metagraph UID/hotkey/last-update widths differ")
    if any(not isinstance(hotkey, str) or not hotkey for hotkey in hotkeys):
        raise ChainWeightStateError("metagraph contains an invalid hotkey")
    if len(set(hotkeys)) != len(hotkeys):
        raise ChainWeightStateError("metagraph contains duplicate hotkeys")
    uids = [_weight_uint(raw, "metagraph UID") for raw in raw_uids]
    last_updates = [
        _weight_uint(raw, "metagraph last-update block") for raw in raw_last_updates
    ]
    if len(set(uids)) != len(uids):
        raise ChainWeightStateError("metagraph contains duplicate UIDs")
    uid_to_hotkey = dict(zip(uids, hotkeys, strict=True))
    try:
        validator_index = hotkeys.index(validator_hotkey)
        validator_uid = uids[validator_index]
    except ValueError:
        return ValidatorWeightSnapshot({}, 0)
    try:
        raw_rows = list(subtensor.weights(netuid=netuid))
    except Exception as exc:
        raise ChainWeightStateError(
            f"cannot fetch validator on-chain weights: {exc}"
        ) from None
    rows: dict[int, dict[int, int]] = {}
    for raw_row in raw_rows:
        if not isinstance(raw_row, (list, tuple)) or len(raw_row) != 2:
            raise ChainWeightStateError("chain weight state contains a malformed row")
        source_uid = _weight_uint(raw_row[0], "chain weight source UID")
        if source_uid not in uid_to_hotkey:
            raise ChainWeightStateError(
                "chain weight state contains a source UID absent from the metagraph"
            )
        if source_uid in rows:
            raise ChainWeightStateError("chain weight state contains duplicate source rows")
        if not isinstance(raw_row[1], (list, tuple)):
            raise ChainWeightStateError("chain weight state contains malformed targets")
        targets: dict[int, int] = {}
        for raw_target in raw_row[1]:
            if not isinstance(raw_target, (list, tuple)) or len(raw_target) != 2:
                raise ChainWeightStateError(
                    "chain weight state contains a malformed target row"
                )
            target_uid = _weight_uint(raw_target[0], "chain weight target UID")
            if target_uid not in uid_to_hotkey:
                raise ChainWeightStateError(
                    "chain weight state contains a target UID absent from the metagraph"
                )
            if target_uid in targets:
                raise ChainWeightStateError(
                    "chain weight state contains duplicate target UIDs"
                )
            targets[target_uid] = _weight_uint(
                raw_target[1], "uint16 weight", maximum=65_535
            )
        rows[source_uid] = targets
    result = {
        uid_to_hotkey[target_uid]: float(weight)
        for target_uid, weight in rows.get(validator_uid, {}).items()
        if weight > 0
    }
    return ValidatorWeightSnapshot(
        normalize(result), last_updates[validator_index]
    )


def read_validator_weights(
    subtensor, netuid: int, validator_hotkey: str
) -> dict[str, float]:
    """Compatibility projection of :func:`read_validator_weight_snapshot`."""

    return read_validator_weight_snapshot(subtensor, netuid, validator_hotkey).weights


# --------------------------------------------------------------------------- #
# RPC wrappers — lazy bittensor; the only code that touches the chain
# --------------------------------------------------------------------------- #

def connect(network: str = "finney", *, fallback_endpoints: Optional[list[str]] = None,
            retry_forever: bool = False):
    """Open a subtensor client. ``network`` is a named network ('finney', 'test') or an
    explicit ``wss://`` endpoint URL. NOTE: the SDK's 'test' alias resolves to
    ``wss://test.finney.opentensor.ai:443`` — pass the URL explicitly if you mean a
    different testnet endpoint. ``fallback_endpoints``/``retry_forever`` enable the
    SDK's retrying substrate client (auto-reconnect through the fallback list)."""
    import bittensor as bt

    kwargs: dict = {}
    if fallback_endpoints:
        kwargs["fallback_endpoints"] = list(fallback_endpoints)
    if retry_forever:
        kwargs["retry_forever"] = True
    return bt.Subtensor(network=network, **kwargs)


def fetch_metagraph(subtensor, netuid: int) -> MetagraphView:
    mg = subtensor.metagraph(netuid=netuid)
    block = int(subtensor.get_current_block())
    return MetagraphView(
        netuid=netuid,
        block=block,
        block_hash=str(subtensor.get_block_hash(block)),  # chain-compat pins get_block_hash
        uids=[int(u) for u in mg.uids],
        hotkeys=list(mg.hotkeys),
        validator_permit=[bool(p) for p in getattr(mg, "validator_permit", [])],
        last_update=[int(value) for value in getattr(mg, "last_update", [])],
    )


def read_commitments(subtensor, netuid: int) -> dict[str, Commitment]:
    """Read every hotkey's current commitment. Optima posts the salted commit hash;
    the reveal (bundle + salt) is verified off-chain by the Ledger."""
    block = int(subtensor.get_current_block())
    raw = subtensor.get_all_commitments(netuid=netuid)  # {hotkey: data}
    out: dict[str, Commitment] = {}
    for hotkey, data in dict(raw).items():
        if data is None:
            continue
        out[hotkey] = Commitment(hotkey=hotkey, data=str(data), block=block)
    return out


def _chain_uint(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ChainRevealHistoryError(f"chain {field} is invalid")
    try:
        result = operator.index(value)
    except (TypeError, OverflowError):
        raise ChainRevealHistoryError(f"chain {field} is invalid") from None
    if result < 0:
        raise ChainRevealHistoryError(f"chain {field} is invalid")
    return result


def _block_hash(subtensor, block: int) -> str:
    try:
        value = str(subtensor.get_block_hash(block))
    except Exception as exc:
        raise ChainRevealHistoryError(
            f"cannot resolve canonical hash for block {block}: {exc}"
        ) from None
    if _BLOCK_HASH_RE.fullmatch(value) is None:
        raise ChainRevealHistoryError(f"chain returned an invalid hash for block {block}")
    return value.lower()


def _finalized_head(subtensor) -> tuple[int, str]:
    """Resolve one exact canonical finalized height/hash pair."""

    direct = getattr(subtensor, "get_finalized_block_number", None)
    if callable(direct):
        try:
            block = _chain_uint(direct(), field="finalized block")
        except ChainRevealHistoryError:
            raise
        except Exception as exc:
            raise ChainRevealHistoryError(
                f"cannot read finalized block number: {exc}"
            ) from None
        return block, _block_hash(subtensor, block)

    substrate = getattr(subtensor, "substrate", None)
    if substrate is None:
        raise ChainRevealHistoryError("subtensor exposes no finalized-chain API")
    try:
        head_hash = str(substrate.get_chain_finalised_head()).lower()
        if _BLOCK_HASH_RE.fullmatch(head_hash) is None:
            raise ValueError("invalid finalized hash")
        number_reader = getattr(substrate, "get_block_number", None)
        if not callable(number_reader):
            raise ValueError("missing finalized block-number resolver")
        block = _chain_uint(number_reader(head_hash), field="finalized block")
    except ChainRevealHistoryError:
        raise
    except Exception as exc:
        raise ChainRevealHistoryError(f"cannot resolve finalized chain head: {exc}") from None
    if _block_hash(subtensor, block) != head_hash:
        raise ChainRevealHistoryError("finalized height/hash pair is inconsistent")
    return block, head_hash


def read_finalized_head(subtensor) -> tuple[int, str]:
    """Return the exact current finalized height/hash for control-plane leases."""

    return _finalized_head(subtensor)


def _unwrap(value: object) -> object:
    seen: set[int] = set()
    while not isinstance(value, (str, bytes, bytearray, dict, list, tuple, int)):
        marker = id(value)
        if marker in seen or not hasattr(value, "value"):
            break
        seen.add(marker)
        value = getattr(value, "value")
    return value


def _event_text(value: object, *, field: str, allow_hex: bool) -> str:
    value = _unwrap(value)
    if isinstance(value, str):
        if allow_hex and value.startswith("0x"):
            try:
                value = bytes.fromhex(value[2:]).decode("utf-8")
            except (UnicodeDecodeError, ValueError):
                raise ChainRevealHistoryError(
                    f"CommitmentRevealed {field} is not UTF-8"
                ) from None
        result = value
    elif allow_hex and isinstance(value, (bytes, bytearray)):
        try:
            result = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            raise ChainRevealHistoryError(
                f"CommitmentRevealed {field} is not UTF-8"
            ) from None
    elif allow_hex and isinstance(value, (list, tuple)) and all(
        type(item) is int and 0 <= item <= 255 for item in value
    ):
        try:
            result = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            raise ChainRevealHistoryError(
                f"CommitmentRevealed {field} is not UTF-8"
            ) from None
    else:
        raise ChainRevealHistoryError(f"CommitmentRevealed {field} is malformed")
    if not result or len(result.encode("utf-8")) > 4096 or "\x00" in result:
        raise ChainRevealHistoryError(f"CommitmentRevealed {field} is malformed")
    return result


def _event_envelope(record: object) -> tuple[str, str, object, int | None]:
    raw = _unwrap(record)
    if not isinstance(raw, dict):
        raise ChainRevealHistoryError("chain event record is not an object")
    event = _unwrap(raw.get("event"))
    if not isinstance(event, dict):
        raise ChainRevealHistoryError("chain event payload is not an object")
    module = event.get("module_id", event.get("module"))
    name = event.get("event_id", event.get("event"))
    if not isinstance(module, str) or not isinstance(name, str):
        raise ChainRevealHistoryError("chain event identity is malformed")
    phase = _unwrap(raw.get("phase"))
    extrinsic_index = None
    if isinstance(phase, dict) and set(phase) == {"ApplyExtrinsic"}:
        extrinsic_index = _chain_uint(
            _unwrap(phase["ApplyExtrinsic"]), field="extrinsic index"
        )
    return module, name, _unwrap(event.get("attributes")), extrinsic_index


def _attribute(
    attributes: dict, names: tuple[str, ...], *, field: str
) -> object:
    present = [name for name in names if name in attributes]
    if len(present) != 1:
        raise ChainRevealHistoryError(
            f"CommitmentRevealed event has ambiguous/missing {field}"
        )
    return _unwrap(attributes[present[0]])


def _reveal_events_at(
    subtensor, *, netuid: int, block: int, block_hash: str
) -> tuple[_RevealEventOccurrence, ...]:
    substrate = getattr(subtensor, "substrate", None)
    get_events = getattr(substrate, "get_events", None)
    if not callable(get_events):
        raise ChainRevealHistoryError("subtensor exposes no finalized event API")
    try:
        records = list(get_events(block_hash=block_hash))
    except Exception as exc:
        raise ChainRevealHistoryError(
            f"cannot read finalized events for block {block}: {exc}"
        ) from None
    rows: list[_RevealEventOccurrence] = []
    for event_index, record in enumerate(records):
        module, name, attributes, extrinsic_index = _event_envelope(record)
        if module != "Commitments" or name != "CommitmentRevealed":
            continue
        if not isinstance(attributes, dict):
            raise ChainRevealHistoryError(
                "CommitmentRevealed attributes are not a named object"
            )
        event_netuid = _chain_uint(
            _attribute(attributes, ("netuid", "net_uid"), field="netuid"),
            field="event netuid",
        )
        if event_netuid != netuid:
            continue
        hotkey = _event_text(
            _attribute(attributes, ("hotkey", "who", "account"), field="hotkey"),
            field="hotkey",
            allow_hex=False,
        )
        rows.append(
            _RevealEventOccurrence(
                hotkey,
                block,
                block_hash,
                event_index,
                extrinsic_index,
            )
        )
    return tuple(rows)


def _storage_history(
    subtensor, netuid: int, *, finalized_block: int, after_block: int | None = None
) -> Counter[tuple[int, str, str]]:
    """Recover complete finalized storage history with bounded backwards pages."""

    observed: Counter[tuple[int, str, str]] = Counter()
    if after_block is not None and (
        type(after_block) is not int or after_block < 0 or after_block > finalized_block
    ):
        raise ChainRevealHistoryError("incremental reveal cursor is invalid")
    query_block = finalized_block
    for _page in range(MAX_REVEAL_HISTORY_PAGES):
        try:
            raw = subtensor.get_all_revealed_commitments(
                netuid=netuid, block=query_block
            )
            page_items = list(dict(raw).items())
        except Exception as exc:
            raise ChainRevealHistoryError(
                "cannot retrieve complete historical reveal state: "
                f"{type(exc).__name__}: {exc}"
            ) from None
        saturated_oldest: list[int] = []
        page_counts: Counter[tuple[int, str, str]] = Counter()
        for hotkey, history in page_items:
            if not history:
                continue
            if (
                not isinstance(hotkey, str)
                or not hotkey
                or len(hotkey) > 256
                or hotkey.strip() != hotkey
            ):
                raise ChainRevealHistoryError("chain reveal history has an invalid hotkey")
            if not isinstance(history, (list, tuple)):
                raise ChainRevealHistoryError("chain reveal history has a malformed row set")
            blocks: list[int] = []
            for entry in history:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise ChainRevealHistoryError("chain reveal history has a malformed row")
                block = _chain_uint(entry[0], field="reveal block")
                data = entry[1]
                if (
                    block > query_block
                    or not isinstance(data, str)
                    or not data
                    or len(data.encode("utf-8")) > 4096
                ):
                    raise ChainRevealHistoryError(
                        "chain reveal history has invalid block/data provenance"
                    )
                if after_block is not None and block <= after_block:
                    continue
                blocks.append(block)
                page_counts[(block, hotkey, data)] += 1
            if after_block is None and len(history) >= CHAIN_REVEAL_HISTORY_CAP:
                oldest = min(blocks)
                saturated_oldest.append(oldest)
        for key, count in page_counts.items():
            observed[key] = max(observed[key], count)
        if len(observed) > MAX_REVEAL_HISTORY_ROWS:
            raise ChainRevealHistoryError("historical reveal row budget exceeded")
        if not saturated_oldest:
            return observed
        boundary = max(saturated_oldest)
        if boundary == 0:
            return observed
        next_block = boundary - 1
        if next_block >= query_block:
            raise ChainRevealHistoryError("historical reveal pagination did not progress")
        query_block = next_block
    raise ChainRevealHistoryError("historical reveal page budget exceeded")


def read_finalized_reveal_history(
    subtensor, netuid: int, *, after_block: int | None = None
) -> FinalizedRevealSnapshot:
    """Reconcile finalized storage rows with canonical event order.

    Storage history proves membership and event records prove same-block ordering.
    Missing/pruned/mismatched events fail closed; network fetch never participates
    in this ordering authority.
    """

    if type(netuid) is not int or netuid < 0:
        raise ValueError("netuid must be a non-negative integer")
    finalized_block, finalized_hash = _finalized_head(subtensor)
    storage = _storage_history(
        subtensor, netuid, finalized_block=finalized_block, after_block=after_block
    )
    blocks = (
        tuple(range(after_block + 1, finalized_block + 1))
        if after_block is not None
        else tuple(sorted({key[0] for key in storage}))
    )
    if len(blocks) > MAX_REVEAL_EVENT_BLOCKS:
        raise ChainRevealHistoryError("finalized reveal event-block budget exceeded")
    events: list[_RevealEventOccurrence] = []
    for block in blocks:
        block_hash = _block_hash(subtensor, block)
        events.extend(
            _reveal_events_at(
                subtensor, netuid=netuid, block=block, block_hash=block_hash
            )
        )
    storage_by_pair: dict[tuple[int, str], list[str]] = {}
    for (block, hotkey, data), count in storage.items():
        storage_by_pair.setdefault((block, hotkey), []).extend([data] * count)
    events_by_pair: dict[tuple[int, str], list[_RevealEventOccurrence]] = {}
    for event in events:
        events_by_pair.setdefault((event.block, event.hotkey), []).append(event)
    storage_counts = Counter(
        {pair: len(payloads) for pair, payloads in storage_by_pair.items()}
    )
    event_counts = Counter(
        {pair: len(occurrences) for pair, occurrences in events_by_pair.items()}
    )
    if storage_counts != event_counts:
        missing = storage_counts - event_counts
        extra = event_counts - storage_counts
        raise ChainRevealHistoryError(
            "finalized reveal storage/event mismatch "
            f"(missing={sum(missing.values())}, extra={sum(extra.values())})"
        )
    ordered_rows: list[RevealedCommitment] = []
    for pair, occurrences in events_by_pair.items():
        # CommitmentRevealed exposes only (netuid, who) on the pinned chain.
        # Event indices order different hotkeys. Multiple occurrences for one
        # hotkey/block are indistinguishable, so payload bytes receive their
        # sole deterministic lexical sub-order within that exact pair.
        payloads = sorted(storage_by_pair[pair])
        occurrences.sort(key=lambda row: row.event_index)
        for occurrence, data in zip(occurrences, payloads, strict=True):
            ordered_rows.append(
                RevealedCommitment(
                    occurrence.hotkey,
                    data,
                    occurrence.block,
                    occurrence.block_hash,
                    occurrence.event_index,
                    occurrence.extrinsic_index,
                )
            )
    ordered = tuple(sorted(ordered_rows, key=lambda row: row.priority_key))
    return FinalizedRevealSnapshot(finalized_block, finalized_hash, ordered)


def read_reveal_history(subtensor, netuid: int) -> tuple[RevealedCommitment, ...]:
    """Compatibility projection of exact finalized history."""

    return read_finalized_reveal_history(subtensor, netuid).reveals


def read_revealed_commitments(subtensor, netuid: int) -> dict[str, RevealedCommitment]:
    """Latest finalized reveal per hotkey, never a head-state authority."""

    result: dict[str, RevealedCommitment] = {}
    for row in read_finalized_reveal_history(subtensor, netuid).reveals:
        previous = result.get(row.hotkey)
        if previous is None or row.priority_key > previous.priority_key:
            result[row.hotkey] = row
    return result


def set_weights(subtensor, wallet, netuid: int, weights_by_hotkey: dict[str, float], *,
                version_key: int = WEIGHTS_VERSION_KEY, dry_run: bool = False,
                wait_for_inclusion: bool = True, wait_for_finalization: bool = False) -> dict:
    """Submit one already-authorized global weight projection on-chain.

    ``dry_run=True`` builds the ``(uids, weights)`` payload from the live metagraph and
    logs it WITHOUT signing or submitting — so the payload can be eyeballed before going
    live. The control-plane publication state machine owns intent and readback.
    """
    mg = fetch_metagraph(subtensor, netuid)
    uids, weights = weights_to_uid_vector(weights_by_hotkey, mg)
    if not uids:
        logger.warning("set_weights: no on-chain hotkeys to weight (champion deregistered?)")
        return {"submitted": False, "reason": "no eligible uids", "uids": [], "weights": []}
    if dry_run:
        logger.info("DRY RUN set_weights netuid=%s version_key=%s uids=%s weights=%s",
                    netuid, version_key, uids, weights)
        return {"submitted": False, "dry_run": True, "uids": uids, "weights": weights}
    result = subtensor.set_weights(
        wallet=wallet, netuid=netuid, uids=uids, weights=weights,
        version_key=version_key, wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    # An included extrinsic can still FAIL chain-side (rate limit, permit, CR
    # window) — report that honestly or the caller records weights that never
    # applied. Measured on 307 (2026-07-10): a second commit 24 blocks after the
    # first was accepted by the SDK but never revealed (weights_rate_limit=100
    # applies to CR commits too); the old unconditional submitted=True wrote the
    # state file and suppressed every retry.
    if isinstance(result, tuple):  # older SDKs: (success, message)
        ok, message = bool(result[0]), str(result[1] if len(result) > 1 else "")
    else:
        ok = bool(getattr(result, "success", result))
        message = str(getattr(result, "message", ""))
    if not ok:
        logger.warning("set_weights failed on-chain: %s", message or result)
    return {"submitted": ok, "result": result, "message": message,
            "uids": uids, "weights": weights}


def post_commitment(subtensor, wallet, netuid: int, data: str, *, dry_run: bool = False) -> dict:
    """Miner side: post a commitment (Optima's salted commit hash) on-chain."""
    if dry_run:
        logger.info("DRY RUN set_commitment netuid=%s data=%s", netuid, data)
        return {"submitted": False, "dry_run": True, "data": data}
    result = subtensor.set_commitment(wallet=wallet, netuid=netuid, data=data)
    return {"submitted": True, "result": result}


def post_reveal_commitment(subtensor, wallet, netuid: int, data: str, *,
                           blocks_until_reveal: int = 10, dry_run: bool = False) -> dict:
    """Miner side: post a timelock-encrypted commitment (the submission payload).

    The payload is drand-encrypted by the SDK and auto-revealed by the chain after
    ``blocks_until_reveal`` blocks — a copycat cannot read the bundle URL before the
    reveal, and the reveal block is the consensus priority timestamp. Hotkey-signed
    (no coldkey needed). Chain-side cap: 1024 bytes; budget ~3100 bytes/hotkey/epoch.
    """
    if dry_run:
        logger.info("DRY RUN set_reveal_commitment netuid=%s bytes=%d data=%s",
                    netuid, len(data.encode("utf-8")), data)
        return {"submitted": False, "dry_run": True, "data": data}
    result = subtensor.set_reveal_commitment(
        wallet=wallet, netuid=netuid, data=data, blocks_until_reveal=blocks_until_reveal,
    )
    return {"submitted": True, "result": result}


def preflight(subtensor, wallet, netuid: int) -> list:
    """Cheap pre-checks before scoring/weighting: is this validator registered, and does
    it hold a validator permit? Returns a list of ``Check`` (reuses the canary's type)."""
    from optima.chain_canary import Check

    checks: list[Check] = []
    hotkey = wallet.hotkey.ss58_address
    registered = bool(subtensor.is_hotkey_registered(hotkey_ss58=hotkey, netuid=netuid))
    checks.append(Check(f"hotkey registered on netuid {netuid}", registered, hotkey))
    if registered:
        mg = fetch_metagraph(subtensor, netuid)
        uid = mg.uid_of(hotkey)
        permit = bool(uid is not None and uid < len(mg.validator_permit)
                      and mg.validator_permit[uid])
        checks.append(Check(
            "validator permit", permit,
            f"uid {uid}" if permit else "no permit — weights ignored until you have stake/permit",
        ))
    return checks
