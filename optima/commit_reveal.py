"""Commit-reveal + king-of-the-hill scoring — the anti-copy mechanism.

The problem in any open competition where submissions are evaluated in the open:
a lazy miner copies the current leader's submission (it's just code shipped to
the validator) and resubmits it, splitting reward for no work. Two mechanisms
defeat that here:

1. **Commit-reveal.** A miner first posts a *commitment* — a hash of
   ``(content_hash, hotkey, salt)`` — during the commit window, before any bundle
   is revealed. Later, in the reveal window, they post ``(content_hash, salt)``.
   A reveal is only accepted if it matches a commitment that *that hotkey* posted
   earlier. So you cannot reveal a bundle you didn't already commit to — and you
   couldn't have committed to a competitor's bundle you hadn't seen yet. Copying
   at reveal time is therefore impossible; the copier has no matching commitment.
   If two miners independently committed to the *same* content, the earliest
   commitment (lowest sequence) is the original; later identical ones are copies
   and earn nothing.

2. **Improvement-over-best (king of the hill).** A standing *champion* (the best
   validated bundle so far) holds the title and the emission. A challenger only
   takes the title if its score beats the champion's by a margin (which absorbs
   measurement noise). A copy ties the champion — it never clears the margin — so
   it earns zero. The only way to earn is to genuinely beat the best.

This module is pure-Python and persists to a JSON ledger so it can be tested and
reasoned about without a GPU. In a real Bittensor subnet the commitments live
on-chain, the bundles are fetched from a content-addressed store, and ``hotkey``
is the miner's SS58 address; the semantics here are the same.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("optima.ledger")

# Bump when the on-disk ledger format changes in a way older code cannot read.
SCHEMA_VERSION = 1


def make_commitment(content_hash: str, hotkey: str, salt: str) -> str:
    """The value a miner posts in the commit window."""
    return hashlib.sha256(f"{content_hash}:{hotkey}:{salt}".encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, data: object) -> None:
    """Write JSON durably: serialize to a sibling temp file, then atomically rename
    it over the target. A crash mid-write leaves the previous file intact — never a
    truncated half-file. ``os.replace`` is atomic on a single filesystem."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _only_fields(cls: type, d: dict) -> dict:
    """Keep only keys that name a field of ``cls``. Unknown keys (written by a newer
    schema) are dropped, and missing keys fall back to the dataclass defaults — so a
    record can gain optional fields without breaking older or newer ledger files."""
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in d.items() if k in names}


def _quarantine(path: Path) -> Path:
    """Move an unreadable ledger aside to ``<name>.corrupt.N`` so a fresh ledger can
    start without silently destroying the damaged file. Returns the new path."""
    for i in range(1, 10_000):
        target = path.with_name(f"{path.name}.corrupt.{i}")
        if not target.exists():
            os.replace(path, target)
            return target
    target = path.with_name(f"{path.name}.corrupt.overflow")
    os.replace(path, target)
    return target


@dataclass
class Commitment:
    hotkey: str
    commitment: str
    round_id: int
    seq: int  # monotonic; commit order = anti-copy priority


@dataclass
class Reveal:
    hotkey: str
    content_hash: str
    salt: str
    round_id: int
    commit_seq: int
    original: bool = True
    fingerprint: str = ""  # reformat-invariant near-copy fingerprint (auto-demotes a match)
    structural_fingerprint: str = ""  # rename/constant-tweak skeleton — ADVISORY only (review)
    # Per-slot reformat-invariant fingerprints (slot -> hash). The LOAD-BEARING copy
    # compare: matching ANY single slot demotes, so padding a stolen bundle with an
    # extra unrelated op cannot perturb the whole-bundle ``fingerprint`` into freshness.
    slot_fingerprints: dict[str, str] = field(default_factory=dict)
    # Per-slot PATH-INDEPENDENT fingerprints of each substantial closure file
    # (slot -> sorted hashes). Catches relocation/padding WITHIN a slot: the ledger
    # demotes on set CONTAINMENT (all of a prior reveal's files appear here), which a
    # stolen body moved into an imported module cannot evade and a merely-shared
    # vendored utility cannot trigger.
    slot_file_fingerprints: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Score:
    hotkey: str
    content_hash: str
    round_id: int
    score: float
    kl_mean: float
    passed: bool
    sglang_version: str = ""  # the pin this speedup was measured against (re-baseline key)
    slot: str = ""  # the slot this submission competes in (for per-slot championships)


@dataclass
class Champion:
    content_hash: str
    hotkey: str
    score: float
    round_id: int
    sglang_version: str = ""  # the pin the title was won under; a different current pin = stale


@dataclass(frozen=True)
class EvalRecord:
    """The typed result of evaluating one bundle — the audit row and the dedup key.
    ``score`` / ``passed`` / ``mean_kl`` mirror the king-of-the-hill ``Score`` atom; the
    rest is the fidelity detail an eval actually produces. Keyed in the ledger by
    ``(hotkey, bundle_hash)`` so an already-scored submission is never re-run. Add a
    field when a producer needs it — ``schema_version`` + the tolerant load make that safe.
    """
    hotkey: str
    bundle_hash: str
    slot: str
    round_id: int
    score: float
    passed: bool
    throughput: float = 0.0
    mean_kl: float = 0.0
    gsm8k_acc: float = -1.0  # -1 = not measured
    dq_reason: str = ""


@dataclass
class SettleResult:
    champion: Optional[Champion]
    weights: dict[str, float]
    title_changed: bool
    challenger_score: float
    rejected_copies: list[str] = field(default_factory=list)  # hotkeys
    champion_stale: bool = False  # champion was crowned under a different sglang pin -> re-baseline


@dataclass
class PerSlotSettleResult:
    """Result of a per-slot championship: one champion PER slot, emission split across
    slots. Pays specialists for the slot they actually own — the fix for winner-take-all
    starving everyone but the single best end-to-end bundle (report misalignment M4)."""
    champions: dict[str, Champion]  # slot -> champion
    weights: dict[str, float]  # hotkey -> emission share (sums to ~1 across slots with a champion)
    title_changes: dict[str, bool]  # slot -> did the title change this round
    stale_slots: list[str] = field(default_factory=list)  # slots whose champion is on an old pin
    rejected_copies: list[str] = field(default_factory=list)


class RevealError(ValueError):
    pass


class Ledger:
    def __init__(self) -> None:
        self.commitments: list[Commitment] = []
        self.reveals: list[Reveal] = []
        self.scores: list[Score] = []
        self.evals: dict[str, EvalRecord] = {}
        self.champion: Optional[Champion] = None  # winner-take-all baseline (single best)
        self.champions: dict[str, Champion] = {}  # per-slot championships (settle_per_slot)
        self._seq = 0

    # ---- persistence ----

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        p = Path(path)
        led = cls()
        if not p.exists():
            return led
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            moved = _quarantine(p)
            logger.warning("ledger %s unreadable (%s); quarantined to %s, starting fresh",
                           p, exc, moved)
            return led
        ver = data.get("schema_version", 1)
        if ver > SCHEMA_VERSION:
            raise ValueError(
                f"ledger {p} is schema v{ver}, newer than this build supports (v{SCHEMA_VERSION}); "
                "upgrade optima before reading it"
            )
        led.commitments = [Commitment(**_only_fields(Commitment, c)) for c in data.get("commitments", [])]
        led.reveals = [Reveal(**_only_fields(Reveal, r)) for r in data.get("reveals", [])]
        led.scores = [Score(**_only_fields(Score, s)) for s in data.get("scores", [])]
        led.evals = {k: EvalRecord(**_only_fields(EvalRecord, v)) for k, v in data.get("evals", {}).items()}
        champ = data.get("champion")
        led.champion = Champion(**_only_fields(Champion, champ)) if champ else None
        led.champions = {
            slot: Champion(**_only_fields(Champion, c))
            for slot, c in (data.get("champions") or {}).items() if c
        }
        led._seq = data.get("seq", len(led.commitments))
        return led

    def save(self, path: str | Path) -> None:
        data = {
            "schema_version": SCHEMA_VERSION,
            "commitments": [asdict(c) for c in self.commitments],
            "reveals": [asdict(r) for r in self.reveals],
            "scores": [asdict(s) for s in self.scores],
            "evals": {k: asdict(v) for k, v in self.evals.items()},
            "champion": asdict(self.champion) if self.champion else None,
            "champions": {slot: asdict(c) for slot, c in self.champions.items()},
            "seq": self._seq,
        }
        _atomic_write_json(Path(path), data)

    # ---- commit phase ----

    def commit(self, hotkey: str, commitment: str, round_id: int) -> int:
        seq = self._seq
        self._seq += 1
        self.commitments.append(Commitment(hotkey, commitment, round_id, seq))
        return seq

    # ---- reveal phase ----

    def reveal(self, hotkey: str, content_hash: str, salt: str, round_id: int,
               fingerprint: str = "", structural_fingerprint: str = "",
               slot_fingerprints: Optional[dict[str, str]] = None,
               slot_file_fingerprints: Optional[dict[str, list[str]]] = None) -> Reveal:
        """Verify a reveal against this hotkey's prior commitments; record it.

        Raises RevealError if no commitment by this hotkey matches. The commitment
        match is per-round (you commit and reveal within a round). Copy detection is
        **cumulative across ALL rounds** and matches on any of:

        * the exact ``content_hash``;
        * the whole-bundle reformat-invariant ``fingerprint``;
        * any single slot of ``slot_fingerprints`` (a stolen slot inside a bundle
          PADDED with an extra op);
        * per-slot file-set CONTAINMENT via ``slot_file_fingerprints`` — every
          substantial closure file of one reveal appearing in the other (a stolen
          body RELOCATED into an imported module, or a slot padded with extra
          files). Containment, not intersection, so two honest miners vendoring
          the same public utility next to their own distinct kernels never match.

        (All from ``optima.copy_fingerprint``.) Earliest commit (lowest seq) by a
        DIFFERENT hotkey is the original; this reveal is a copy if such an earlier
        one exists.
        """
        target = make_commitment(content_hash, hotkey, salt)
        match = min(
            (c for c in self.commitments
             if c.hotkey == hotkey and c.round_id == round_id and c.commitment == target),
            key=lambda c: c.seq,
            default=None,
        )
        if match is None:
            raise RevealError(
                f"no commitment by {hotkey!r} in round {round_id} matches the revealed bundle"
            )

        # Copy detection: a DIFFERENT hotkey's earlier reveal of the same content
        # (exact hash) OR the same normalized structure (near-copy fingerprint), in
        # ANY round, makes the later commit the copy. Same-hotkey re-reveals of one's
        # own work are never copies. Earliest commit_seq wins.
        slot_fps = {s: fp for s, fp in (slot_fingerprints or {}).items() if fp}
        file_fps = {s: sorted(v) for s, v in (slot_file_fingerprints or {}).items() if v}

        def _same(r: Reveal) -> bool:
            if r.hotkey == hotkey:
                return False
            if r.content_hash == content_hash:
                return True
            if fingerprint and r.fingerprint == fingerprint:
                return True
            # Per-slot compare: one stolen slot demotes, however the rest of the
            # bundle was padded/perturbed.
            if any(r.slot_fingerprints.get(s) == fp for s, fp in slot_fps.items()):
                return True
            # Per-slot file-set CONTAINMENT (either direction — commit order decides
            # who is original): all of one bundle's substantial files for a slot
            # appearing inside the other's = the same work, wherever the copier
            # relocated it and whatever they padded around it.
            for s, mine in file_fps.items():
                theirs = set(r.slot_file_fingerprints.get(s, ()))
                if theirs and (theirs <= set(mine) or set(mine) <= theirs):
                    return True
            return False

        prior = [r for r in self.reveals if _same(r)]
        original = all(match.seq < r.commit_seq for r in prior) if prior else True
        if prior and original:
            # This reveal predates earlier-recorded ones; demote them.
            for r in prior:
                r.original = False

        rev = Reveal(hotkey, content_hash, salt, round_id, match.seq, original,
                     fingerprint, structural_fingerprint, slot_fps, file_fps)
        self.reveals.append(rev)
        return rev

    # ---- emission policy ----

    def current_weights(self, per_slot: bool = True) -> dict[str, float]:
        """The emission weights implied by the CURRENT champion state (no re-settle).

        THE single swap point for emission policy: every weight consumer (the chain
        validator loop, ``optima set-weights``) reads this instead of re-deriving
        winner-take-all inline. Today: per-slot championships split emission equally
        across slots (a hotkey holding k of n slots earns k/n); ``per_slot=False``
        is the single-champion baseline. The planned relative-improvement +
        time-decay scheme replaces THIS method's body, nothing else.
        """
        if per_slot and self.champions:
            share = 1.0 / len(self.champions)
            weights: dict[str, float] = {}
            for champ in self.champions.values():
                weights[champ.hotkey] = weights.get(champ.hotkey, 0.0) + share
            return weights
        if self.champion:
            return {self.champion.hotkey: 1.0}
        return {}

    def structural_near_copies(self, structural_fingerprint: str, hotkey: str) -> list[str]:
        """ADVISORY: prior reveals by OTHER hotkeys whose structural skeleton matches
        (rename/constant-tweak similarity). Returned for review/flagging — NOT used to
        demote, since the skeleton can collide on genuinely-distinct simple kernels."""
        if not structural_fingerprint:
            return []
        return sorted({
            r.hotkey for r in self.reveals
            if r.hotkey != hotkey and r.structural_fingerprint == structural_fingerprint
        })

    # ---- scoring ----

    def record_score(self, hotkey: str, content_hash: str, round_id: int,
                     score: float, kl_mean: float, passed: bool, sglang_version: str = "",
                     slot: str = "") -> None:
        self.scores.append(Score(hotkey, content_hash, round_id, score, kl_mean, passed,
                                 sglang_version, slot))

    # ---- full eval records (audit trail + dedup; the rich superset of a Score) ----

    @staticmethod
    def _eval_key(hotkey: str, bundle_hash: str) -> str:
        return f"{hotkey}:{bundle_hash}"

    def record_eval(self, rec: EvalRecord) -> None:
        """Store the full eval record, keyed by (hotkey, bundle_hash). Recording the
        same submission again overwrites it (evaluations are deterministic)."""
        self.evals[self._eval_key(rec.hotkey, rec.bundle_hash)] = rec

    def is_known(self, hotkey: str, bundle_hash: str) -> bool:
        """True if this exact submission already has an eval record — skip re-running it."""
        return self._eval_key(hotkey, bundle_hash) in self.evals

    def eval_for(self, hotkey: str, bundle_hash: str) -> Optional[EvalRecord]:
        return self.evals.get(self._eval_key(hotkey, bundle_hash))

    def _is_original(self, hotkey: str, content_hash: str, round_id: int) -> bool:
        for r in self.reveals:
            if r.hotkey == hotkey and r.content_hash == content_hash and r.round_id == round_id:
                return r.original
        return False

    def settle(self, round_id: int, margin: float = 0.02,
               current_sglang_version: str = "") -> SettleResult:
        """Apply king-of-the-hill: a challenger takes the title only if it beats the
        champion by ``margin``. Emission goes to the champion (winner-take-all baseline).
        Copies and non-improvers earn nothing.

        The recorded ``score`` is already a NOISE-CONFIRMED crownable speedup vs the
        round's fresh stock baseline, or 0.0 (see the eval) — so a too-noisy or
        below-bar candidate cannot win here either.

        STALE CHAMPION: a champion's frozen ``score`` is a speedup vs the stock kernels
        of the pin it was crowned under. After a ``PINNED_SGLANG`` bump the stock baseline
        changes, so that frozen number is no longer comparable to a challenger measured
        against the NEW stock. When ``current_sglang_version`` differs from the champion's,
        we refuse to let the stale number gate the round: the best confident challenger
        re-establishes the title by clearing the floor margin over *current* stock, and
        ``champion_stale`` is flagged so the operator re-baselines the old champion.
        """
        rejected_copies: list[str] = []
        candidates: list[Score] = []
        for s in self.scores:
            if s.round_id != round_id or not s.passed:
                continue
            if not self._is_original(s.hotkey, s.content_hash, round_id):
                rejected_copies.append(s.hotkey)
                continue
            candidates.append(s)

        challenger = max(candidates, key=lambda s: s.score, default=None)
        challenger_score = challenger.score if challenger else 0.0

        champion_stale = bool(
            self.champion and current_sglang_version and self.champion.sglang_version
            and self.champion.sglang_version != current_sglang_version
        )
        # A stale champion's frozen ratio isn't comparable to the current pin's baseline,
        # so don't gate on it — require a real win over current fresh stock instead.
        if self.champion and not champion_stale:
            threshold = self.champion.score * (1.0 + margin)
        else:
            threshold = 1.0 + margin

        title_changed = False
        if challenger is not None and challenger_score >= threshold:
            self.champion = Champion(
                content_hash=challenger.content_hash,
                hotkey=challenger.hotkey,
                score=challenger.score,
                round_id=round_id,
                sglang_version=current_sglang_version or challenger.sglang_version,
            )
            title_changed = True
            champion_stale = False  # freshly (re-)crowned under the current pin

        weights = {self.champion.hotkey: 1.0} if self.champion else {}
        return SettleResult(
            champion=self.champion,
            weights=weights,
            title_changed=title_changed,
            challenger_score=challenger_score,
            rejected_copies=sorted(set(rejected_copies)),
            champion_stale=champion_stale,
        )

    def settle_per_slot(self, round_id: int, margin: float = 0.02,
                        current_sglang_version: str = "") -> PerSlotSettleResult:
        """Per-slot king-of-the-hill: a champion PER slot, emission split equally across
        the slots that have a champion. This pays a specialist who owns ONE slot, instead
        of giving 100% to the single best end-to-end bundle (winner-take-all starves
        everyone else — report misalignment M4). Same noise-confirmed crownable scores,
        same copy exclusion, same stale-on-pin-bump handling as ``settle`` — applied
        within each slot's bracket. Updates ``self.champions`` (the per-slot map).
        """
        rejected_copies: list[str] = []
        by_slot: dict[str, list[Score]] = {}
        for s in self.scores:
            if s.round_id != round_id or not s.passed:
                continue
            if not self._is_original(s.hotkey, s.content_hash, round_id):
                rejected_copies.append(s.hotkey)
                continue
            by_slot.setdefault(s.slot, []).append(s)

        title_changes: dict[str, bool] = {}
        stale_slots: list[str] = []
        for slot, cands in by_slot.items():
            challenger = max(cands, key=lambda s: s.score, default=None)
            if challenger is None:
                continue
            champ = self.champions.get(slot)
            stale = bool(champ and current_sglang_version and champ.sglang_version
                         and champ.sglang_version != current_sglang_version)
            threshold = (champ.score * (1.0 + margin)) if (champ and not stale) else (1.0 + margin)
            if challenger.score >= threshold:
                self.champions[slot] = Champion(
                    content_hash=challenger.content_hash, hotkey=challenger.hotkey,
                    score=challenger.score, round_id=round_id,
                    sglang_version=current_sglang_version or challenger.sglang_version,
                )
                title_changes[slot] = True
            elif stale:
                stale_slots.append(slot)

        # A slot with NO submissions this round still has a standing champion earning
        # emission; if that champion was crowned under a different pin it must be
        # flagged for re-baseline too — staleness is a property of the champion, not
        # of this round's challenger traffic.
        for slot, champ in self.champions.items():
            if slot in by_slot or not champ:
                continue
            if (current_sglang_version and champ.sglang_version
                    and champ.sglang_version != current_sglang_version):
                stale_slots.append(slot)

        # Split emission equally across slots that currently have a champion.
        live = {slot: c for slot, c in self.champions.items() if c}
        weights: dict[str, float] = {}
        if live:
            share = 1.0 / len(live)
            for c in live.values():
                weights[c.hotkey] = weights.get(c.hotkey, 0.0) + share
        return PerSlotSettleResult(
            champions=dict(self.champions),
            weights=weights,
            title_changes=title_changes,
            stale_slots=sorted(set(stale_slots)),
            rejected_copies=sorted(set(rejected_copies)),
        )
