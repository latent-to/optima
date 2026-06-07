"""On-chain I/O: read commitments, push king-of-the-hill weights, preflight checks.

Architecture mirrors the sglang seam. **Pure helpers** (weight-vector math, hotkey↔uid
mapping) carry no SDK and no network, so they are unit-tested directly. **Thin RPC
wrappers** lazily ``import bittensor`` and are the only code that touches the chain —
so the package still imports, and the test suite still runs, with no SDK installed.

The exact SDK methods called here are pinned by ``optima chain-compat``
(optima/chain_canary.py); run it after any bittensor bump, and the wrappers' calls
stay a thin, auditable layer over what that canary asserts.

Optima keeps its own commit-reveal: a miner posts the *salted commit hash* via
``set_commitment``; the validator reads it via ``get_all_commitments``; the reveal
(bundle + salt) is verified off-chain by the Ledger. The chain is the durable,
consensus source of *what was committed* and *who won*; the Ledger is the scoring half.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("optima.chain")

# Yuma-consensus version stamped on set_weights. A coordinated subnet parameter,
# bumped deliberately (like PINNED_SGLANG) so every validator agrees.
WEIGHTS_VERSION_KEY = 1


@dataclass
class Commitment:
    """A hotkey's current on-chain commitment — for Optima, the salted commit hash."""
    hotkey: str
    data: str
    block: int = 0


@dataclass
class MetagraphView:
    """A minimal, SDK-free snapshot of the metagraph used for weight-setting + seeding."""
    netuid: int
    block: int
    block_hash: str
    uids: list[int] = field(default_factory=list)
    hotkeys: list[str] = field(default_factory=list)  # index-aligned with uids
    validator_permit: list[bool] = field(default_factory=list)

    def uid_of(self, hotkey: str) -> Optional[int]:
        try:
            return self.uids[self.hotkeys.index(hotkey)]
        except ValueError:
            return None


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
    *live* metagraph. Hotkeys absent from the metagraph (deregistered since the eval)
    are dropped; the remainder is renormalized to sum 1.0."""
    on_chain = {hk: w for hk, w in weights_by_hotkey.items()
                if metagraph.uid_of(hk) is not None}
    norm = normalize(on_chain)
    uids = [metagraph.uid_of(hk) for hk in norm]
    weights = [norm[hk] for hk in norm]
    return uids, weights


# --------------------------------------------------------------------------- #
# RPC wrappers — lazy bittensor; the only code that touches the chain
# --------------------------------------------------------------------------- #

def connect(network: str = "finney"):
    """Open a subtensor client. ``network='test'`` for the public testnet."""
    import bittensor as bt

    return bt.Subtensor(network=network)


def _block_hash(subtensor, block: int) -> str:
    """Best-effort block hash for prompt seeding. Falls back to the block number (still
    consensus-deterministic) if the SDK exposes no hash accessor."""
    for name in ("get_block_hash", "block_hash"):
        fn = getattr(subtensor, name, None)
        if callable(fn):
            try:
                return str(fn(block))
            except Exception:  # noqa: BLE001
                pass
    return str(block)


def fetch_metagraph(subtensor, netuid: int) -> MetagraphView:
    mg = subtensor.metagraph(netuid=netuid)
    block = int(subtensor.get_current_block())
    return MetagraphView(
        netuid=netuid,
        block=block,
        block_hash=_block_hash(subtensor, block),
        uids=[int(u) for u in mg.uids],
        hotkeys=list(mg.hotkeys),
        validator_permit=[bool(p) for p in getattr(mg, "validator_permit", [])],
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


def set_weights(subtensor, wallet, netuid: int, weights_by_hotkey: dict[str, float], *,
                version_key: int = WEIGHTS_VERSION_KEY, dry_run: bool = False,
                wait_for_inclusion: bool = True, wait_for_finalization: bool = False) -> dict:
    """Push the king-of-the-hill weights on-chain.

    ``dry_run=True`` builds the ``(uids, weights)`` payload from the live metagraph and
    logs it WITHOUT signing or submitting — so the payload can be eyeballed before going
    live. Returns a structured result either way (never raises on an empty champion).
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
    return {"submitted": True, "result": result, "uids": uids, "weights": weights}


def post_commitment(subtensor, wallet, netuid: int, data: str, *, dry_run: bool = False) -> dict:
    """Miner side: post a commitment (Optima's salted commit hash) on-chain."""
    if dry_run:
        logger.info("DRY RUN set_commitment netuid=%s data=%s", netuid, data)
        return {"submitted": False, "dry_run": True, "data": data}
    result = subtensor.set_commitment(wallet=wallet, netuid=netuid, data=data)
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
