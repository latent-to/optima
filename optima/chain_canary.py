"""Bittensor SDK canary — does the installed chain SDK expose the API we depend on?

Optima's chain layer (commitments read, weight setting, metagraph, registration
preflight) calls into the bittensor SDK. The SDK's surface drifts across releases,
so — exactly like the sglang seam canary (``optima compat``) — we introspect the
*installed* package: imports + class attributes + signatures, **no network, no
wallet, no chain connection**. Two jobs:

  1. **Catch a missing API before it matters.** If a method we plan to call isn't on
     the installed SDK, the canary goes red here, offline, instead of failing live
     against the chain.
  2. **Discovery.** The chain SDK renames things between versions; the canary prints
     the actual weight / commitment / metagraph members it finds, so the chain code
     is written against the real names rather than a guess.

Run ``optima chain-compat``. Green means the SDK has what we expect; red lists what
is missing and, via the discovery lines, what is actually there.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


# The subtensor methods Optima's chain layer intends to call. We check the class
# (not an instance), so no chain connection is needed. Names are our *expectation*;
# the discovery checks below report what the installed SDK actually exposes, so a
# rename shows up as a red check next to the real name.
_EXPECTED_SUBTENSOR_METHODS: tuple[tuple[str, str], ...] = (
    ("set_weights", "push king-of-the-hill weights on-chain"),
    ("metagraph", "read uids / hotkeys / stake / validator_permit"),
    ("get_all_commitments", "read every hotkey's commitment (the salted commit hash)"),
    ("set_commitment", "miner posts a commitment on-chain"),
    ("is_hotkey_registered", "preflight: this validator is registered"),
    ("get_current_block", "current block height"),
    ("get_block_hash", "block hash -> prompt seed (consensus + anti-prebake)"),
)


def run_checks() -> list[Check]:
    checks: list[Check] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append(Check(name, bool(ok), str(detail)))

    try:
        import bittensor as bt
    except Exception as exc:  # noqa: BLE001 - SDK absent / broken install
        add("import bittensor", False, repr(exc))
        return checks

    ver = getattr(bt, "__version__", "?")
    add("bittensor installed", True, f"version {ver}")

    # Wallet class (we sign extrinsics + set weights with a wallet). Prefer the
    # capitalized `Wallet` (the live class); `wallet` is a deprecated lowercase alias.
    wallet_cls = getattr(bt, "Wallet", None) or getattr(bt, "wallet", None)
    add("Wallet class present", wallet_cls is not None,
        "bt.Wallet — signs extrinsics")

    # The chain client class. Prefer the capitalized `Subtensor` (the live class in
    # current bittensor); `subtensor` is a deprecated lowercase alias.
    subtensor_cls = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor", None)
    add("Subtensor class present", subtensor_cls is not None,
        "bt.Subtensor — the chain client")
    if subtensor_cls is None:
        return checks

    # The specific methods we plan to call (introspected on the class — no network).
    for method, why in _EXPECTED_SUBTENSOR_METHODS:
        fn = getattr(subtensor_cls, method, None)
        if fn is None:
            add(f"subtensor.{method}", False, f"MISSING — {why}")
            continue
        try:
            sig = f"({', '.join(inspect.signature(fn).parameters)})"
        except (ValueError, TypeError):
            sig = "(signature unavailable)"
        add(f"subtensor.{method}", True, f"{why}; sig {sig}")

    # Discovery: the commitment / weight / metagraph member families. Names vary by
    # release; printing what is actually present lets the chain code target the real
    # API instead of a hard-coded guess.
    def _members(predicate) -> list[str]:
        return sorted(m for m in dir(subtensor_cls) if predicate(m.lower()))

    commit_members = _members(lambda m: "commit" in m or "reveal" in m)
    add("commitment/reveal API present", bool(commit_members),
        f"found: {commit_members or 'NONE'}")

    weight_members = _members(lambda m: "weight" in m)
    add("weights API present", bool(weight_members), f"found: {weight_members or 'NONE'}")

    return checks


def format_checks(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        mark = "ok  " if c.ok else "FAIL"
        lines.append(f"  [{mark}] {c.name}" + (f"  — {c.detail}" if c.detail else ""))
    n_fail = sum(1 for c in checks if not c.ok)
    lines.append("")
    lines.append(
        "CHAIN SDK API PRESENT" if n_fail == 0
        else f"{n_fail} CHECK(S) FAILED — the installed bittensor lacks an API we plan to "
             "use; the discovery lines above show the real member names to target"
    )
    return "\n".join(lines)
