"""Durable finalized-arrival authority, separate from evaluation and settlement."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import stat
import fcntl
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

from optima.copy_fingerprint import (
    SubmittedDeltaFingerprint, compare_submitted_deltas,
)
from optima.eval.evidence_store import EvidenceArtifactRef
from optima.stack_identity import canonical_digest, require_sha256_hex

if TYPE_CHECKING:
    from optima.chain.weights import WeightProjection, WeightPublicationRecord
    from optima.settlement import SettlementCandidate, SettlementPlan
    from optima.stack_manifest import EvaluationStackManifest


_HASH = re.compile(r"[0-9a-f]{64}\Z")
_BLOCK_HASH = re.compile(r"0x[0-9a-f]{64}\Z")
_ACTIVE = ("reserved", "fetching", "transport_retry", "published", "qualifying")
_TERMINAL = ("failed", "expired", "qualified")
_STATUSES = frozenset((*_ACTIVE, *_TERMINAL, "held", "no_decision"))


class IntakeError(RuntimeError):
    """Finalized arrival state is malformed, stale, or unsafe to advance."""


@dataclass(frozen=True)
class IntakeScope:
    genesis_hash: str
    netuid: int

    def __post_init__(self) -> None:
        if _BLOCK_HASH.fullmatch(self.genesis_hash or "") is None:
            raise IntakeError("intake genesis hash is malformed")
        if type(self.netuid) is not int or self.netuid < 0:
            raise IntakeError("intake netuid is malformed")

    def to_dict(self) -> dict[str, object]:
        return {"genesis_hash": self.genesis_hash, "netuid": self.netuid}

    @property
    def digest(self) -> str:
        return canonical_digest("optima.chain.intake-scope", self.to_dict())


@dataclass(frozen=True)
class IntakePolicy:
    epoch_blocks: int = 360
    cutoff_blocks: int = 30
    max_pending: int = 256
    max_per_hotkey_epoch: int = 16
    max_per_target_epoch: int = 64
    max_transport_retries: int = 3
    max_qualification_retries: int = 3
    max_cohort: int = 8
    expiry_blocks: int = 2_880

    def __post_init__(self) -> None:
        values = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        if any(type(value) is not int or value <= 0 for value in values):
            raise IntakeError("intake policy bounds must be positive integers")
        if self.cutoff_blocks >= self.epoch_blocks:
            raise IntakeError("intake cutoff must be smaller than its epoch")
        if self.max_cohort > self.max_pending:
            raise IntakeError("cohort bound exceeds the pending queue bound")


@dataclass(frozen=True)
class FinalizedArrival:
    hotkey: str
    content_hash: str
    url: str
    block: int
    block_hash: str
    event_index: int
    event_subindex: int = 0
    payload_digest: str = ""
    invalid_reason: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.hotkey, str)
            or not self.hotkey
            or self.hotkey.strip() != self.hotkey
            or len(self.hotkey) > 256
            or any(char in self.hotkey for char in "\x00\r\n")
        ):
            raise IntakeError("arrival hotkey is malformed")
        valid_reference = (
            isinstance(self.content_hash, str)
            and _HASH.fullmatch(self.content_hash) is not None
            and isinstance(self.url, str)
            and bool(self.url)
            and not self.invalid_reason
        )
        invalid_reference = (
            self.content_hash == ""
            and self.url == ""
            and isinstance(self.invalid_reason, str)
            and bool(self.invalid_reason)
            and len(self.invalid_reason) <= 2_048
        )
        if not (valid_reference or invalid_reference):
            raise IntakeError("arrival payload disposition is malformed")
        if type(self.block) is not int or self.block < 0:
            raise IntakeError("arrival block is malformed")
        if not isinstance(self.block_hash, str) or _BLOCK_HASH.fullmatch(self.block_hash) is None:
            raise IntakeError("arrival block hash is malformed")
        for field in ("event_index", "event_subindex"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 0:
                raise IntakeError(f"arrival {field} is malformed")
        payload_digest = self.payload_digest or canonical_digest(
            "optima.chain.finalized-payload",
            {"content_hash": self.content_hash, "url": self.url},
        )
        require_sha256_hex(payload_digest, field="payload_digest")
        object.__setattr__(self, "payload_digest", payload_digest)

    @property
    def valid(self) -> bool:
        return not self.invalid_reason

    @property
    def arrival_key(self) -> tuple[int, int, int, str, str]:
        return (
            self.block,
            self.event_index,
            self.event_subindex,
            self.hotkey,
            self.content_hash,
        )

    @property
    def reservation_id(self) -> str:
        return canonical_digest(
            "optima.chain.finalized-arrival",
            {
                "block": self.block,
                "block_hash": self.block_hash,
                "content_hash": self.content_hash,
                "event_index": self.event_index,
                "event_subindex": self.event_subindex,
                "hotkey": self.hotkey,
                "payload_digest": self.payload_digest,
                "url": self.url,
            },
        )


@dataclass(frozen=True)
class IntakeReservation:
    reservation_id: str
    arrival: FinalizedArrival
    admission_epoch: int
    status: str
    target_id: str
    target_members: tuple[str, ...]
    delta_fingerprint: SubmittedDeltaFingerprint | None
    transport_attempts: int
    publication_digest: str
    publication_root: str
    qualification_authority_digest: str
    qualification_evidence_digest: str
    decision: str
    reason: str

    def __post_init__(self) -> None:
        require_sha256_hex(self.reservation_id, field="reservation_id")
        if self.reservation_id != self.arrival.reservation_id:
            raise IntakeError("reservation identity differs from finalized arrival")
        if type(self.admission_epoch) is not int or self.admission_epoch < 0:
            raise IntakeError("reservation epoch is malformed")
        if self.status not in _STATUSES:
            raise IntakeError("reservation status is unsupported")
        if tuple(self.target_members) != tuple(sorted(set(self.target_members))):
            raise IntakeError("reservation target members are not canonical")
        if self.delta_fingerprint is not None and (
            type(self.delta_fingerprint) is not SubmittedDeltaFingerprint
            or self.delta_fingerprint.target_id != self.target_id
            or self.delta_fingerprint.members != self.target_members
        ):
            raise IntakeError("reservation delta fingerprint differs from its target")
        if type(self.transport_attempts) is not int or self.transport_attempts < 0:
            raise IntakeError("reservation transport attempts are malformed")
        for field in (
            "publication_digest",
            "qualification_authority_digest",
            "qualification_evidence_digest",
        ):
            value = getattr(self, field)
            if value and _HASH.fullmatch(value) is None:
                raise IntakeError(f"reservation {field} is malformed")
        if self.decision not in {"", "PASS", "FAIL", "NO_DECISION"}:
            raise IntakeError("reservation decision is unsupported")


@dataclass(frozen=True)
class EvaluationStackState:
    arena_digest: str
    generation: int
    manifest: EvaluationStackManifest
    tree_digest: str
    transition_event_id: str

    def __post_init__(self) -> None:
        from optima.stack_manifest import EvaluationStackManifest

        require_sha256_hex(self.arena_digest, field="arena_digest")
        require_sha256_hex(self.tree_digest, field="tree_digest")
        require_sha256_hex(self.transition_event_id, field="transition_event_id")
        if (
            type(self.generation) is not int
            or self.generation < 0
            or type(self.manifest) is not EvaluationStackManifest
            or self.manifest.arena_digest != self.arena_digest
        ):
            raise IntakeError("evaluation stack state is malformed")

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.chain.evaluation-stack-state",
            {
                "arena_digest": self.arena_digest,
                "generation": self.generation,
                "stack_digest": self.manifest.digest,
                "tree_digest": self.tree_digest,
                "transition_event_id": self.transition_event_id,
            },
        )


@dataclass(frozen=True)
class SettlementLease:
    lease_id: str
    authority_digest: str
    generation: int
    expires_block: int
    stack: EvaluationStackState
    candidates: tuple[SettlementCandidate, ...]
    initial_event_sequence: int
    previous_event_digest: str

    def __post_init__(self) -> None:
        from optima.settlement import SettlementCandidate

        for field in ("lease_id", "authority_digest"):
            require_sha256_hex(getattr(self, field), field=field)
        if (
            type(self.generation) is not int
            or self.generation <= 0
            or type(self.expires_block) is not int
            or self.expires_block <= 0
            or type(self.stack) is not EvaluationStackState
            or type(self.initial_event_sequence) is not int
            or self.initial_event_sequence < 0
        ):
            raise IntakeError("settlement lease bounds are malformed")
        require_sha256_hex(
            self.previous_event_digest,
            field="previous_event_digest",
        ) if self.previous_event_digest else None
        candidates = tuple(self.candidates)
        if (
            not candidates
            or any(type(row) is not SettlementCandidate for row in candidates)
            or any(row.arena_digest != self.stack.arena_digest for row in candidates)
            or any(row.qualification_authority_digest != self.authority_digest for row in candidates)
        ):
            raise IntakeError("settlement lease candidates are inconsistent")
        object.__setattr__(self, "candidates", candidates)


class FinalizedIntakeStore:
    """Single SQLite authority for arrival order, admission, and qualification state."""

    def __init__(
        self,
        path: str | Path,
        policy: IntakePolicy = IntakePolicy(),
        *,
        scope: IntakeScope,
    ):
        if type(policy) is not IntakePolicy:
            raise IntakeError("intake store requires an exact IntakePolicy")
        if type(scope) is not IntakeScope:
            raise IntakeError("intake store requires an exact chain scope")
        requested = Path(path).expanduser()
        if requested.is_symlink():
            raise IntakeError("intake database path must not be a symlink")
        parent_existed = requested.parent.exists()
        requested.parent.mkdir(parents=True, exist_ok=True)
        if not parent_existed:
            os.chmod(requested.parent, 0o700)
        try:
            parent_before = requested.parent.lstat()
            parent = requested.parent.resolve(strict=True)
            parent_after = parent.lstat()
        except OSError as exc:
            raise IntakeError(f"intake database parent is unavailable: {exc}") from None
        if (
            stat.S_ISLNK(parent_before.st_mode)
            or not stat.S_ISDIR(parent_before.st_mode)
            or (parent_before.st_dev, parent_before.st_ino)
            != (parent_after.st_dev, parent_after.st_ino)
            or stat.S_IMODE(parent_after.st_mode) != 0o700
            or (hasattr(os, "geteuid") and parent_after.st_uid != os.geteuid())
        ):
            raise IntakeError(
                "intake database parent must be validator-owned mode 0700"
            )
        self.path = parent / requested.name
        self.policy = policy
        self.scope = scope
        if self.path.exists():
            info = self.path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600
                or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
            ):
                raise IntakeError("existing intake database has unsafe ownership or mode")
        previous_umask = os.umask(0o077)
        try:
            lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
            lock_flags |= getattr(os, "O_NOFOLLOW", 0)
            self._lock_fd = os.open(str(self.path) + ".lock", lock_flags, 0o600)
            lock_info = os.fstat(self._lock_fd)
            if (
                not stat.S_ISREG(lock_info.st_mode)
                or lock_info.st_nlink != 1
                or stat.S_IMODE(lock_info.st_mode) != 0o600
                or lock_info.st_uid != os.geteuid()
            ):
                raise IntakeError("intake controller lock has an unsafe shape")
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise IntakeError("another intake controller owns this database") from None
            self._db = sqlite3.connect(self.path, isolation_level=None, timeout=30.0)
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._create_schema()
            self._bind_scope()
        except Exception:
            if hasattr(self, "_db"):
                self._db.close()
            if hasattr(self, "_lock_fd"):
                os.close(self._lock_fd)
            raise
        finally:
            os.umask(previous_umask)
        os.chmod(self.path, 0o600)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(self.path) + suffix)
            if sidecar.exists():
                os.chmod(sidecar, 0o600)
        self._recover_interrupted()

    def __enter__(self) -> "FinalizedIntakeStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()
        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        os.close(self._lock_fd)

    def _create_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) STRICT;
            CREATE TABLE IF NOT EXISTS reservations (
                reservation_id TEXT PRIMARY KEY,
                block INTEGER NOT NULL,
                block_hash TEXT NOT NULL,
                event_index INTEGER NOT NULL,
                event_subindex INTEGER NOT NULL,
                hotkey TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                url TEXT NOT NULL,
                payload_digest TEXT NOT NULL,
                invalid_reason TEXT NOT NULL DEFAULT '',
                admission_epoch INTEGER NOT NULL,
                status TEXT NOT NULL,
                target_id TEXT NOT NULL DEFAULT '',
                target_members_json TEXT NOT NULL DEFAULT '[]',
                delta_fingerprint_json TEXT NOT NULL DEFAULT '',
                transport_attempts INTEGER NOT NULL DEFAULT 0,
                publication_digest TEXT NOT NULL DEFAULT '',
                publication_root TEXT NOT NULL DEFAULT '',
                qualification_authority_digest TEXT NOT NULL DEFAULT '',
                qualification_authority_json TEXT NOT NULL DEFAULT '',
                qualification_evidence_digest TEXT NOT NULL DEFAULT '',
                retry_group_digest TEXT NOT NULL DEFAULT '',
                retry_position INTEGER NOT NULL DEFAULT 0,
                decision TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                UNIQUE(block_hash, event_index, event_subindex, hotkey, content_hash)
            ) STRICT;
            CREATE INDEX IF NOT EXISTS reservations_order
                ON reservations(block, event_index, event_subindex, hotkey, content_hash);
            CREATE INDEX IF NOT EXISTS reservations_status
                ON reservations(status, admission_epoch, block, event_index, event_subindex);
            CREATE TABLE IF NOT EXISTS qualification_dispositions (
                reservation_id TEXT NOT NULL REFERENCES reservations(reservation_id),
                attempt_index INTEGER NOT NULL,
                authority_digest TEXT NOT NULL,
                authority_manifest_json TEXT NOT NULL,
                evidence_digest TEXT NOT NULL,
                attempt_ref_json TEXT NOT NULL,
                report_digest TEXT NOT NULL,
                failure_digest TEXT NOT NULL,
                decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                PRIMARY KEY(reservation_id, attempt_index)
            ) STRICT;
            CREATE TABLE IF NOT EXISTS settlement_candidates (
                reservation_id TEXT PRIMARY KEY REFERENCES reservations(reservation_id),
                authority_digest TEXT NOT NULL,
                candidate_digest TEXT NOT NULL UNIQUE,
                candidate_json TEXT NOT NULL,
                evidence_root TEXT NOT NULL,
                settlement_evidence_digest TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                lease_id TEXT NOT NULL DEFAULT '',
                lease_generation INTEGER NOT NULL DEFAULT 0,
                lease_expires_block INTEGER NOT NULL DEFAULT 0,
                reason TEXT NOT NULL DEFAULT ''
            ) STRICT;
            CREATE INDEX IF NOT EXISTS settlement_candidates_status
                ON settlement_candidates(status, authority_digest, reservation_id);
            CREATE TABLE IF NOT EXISTS settlement_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL,
                reservation_id TEXT NOT NULL,
                arena_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                event_digest TEXT NOT NULL,
                event_json TEXT NOT NULL
            ) STRICT;
            CREATE TABLE IF NOT EXISTS evaluation_stacks (
                arena_id TEXT PRIMARY KEY,
                generation INTEGER NOT NULL,
                stack_digest TEXT NOT NULL,
                tree_digest TEXT NOT NULL,
                stack_json TEXT NOT NULL,
                transition_event_id TEXT NOT NULL
            ) STRICT;
            CREATE TABLE IF NOT EXISTS standing_reward_claims (
                arena_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                claim_digest TEXT NOT NULL UNIQUE,
                claim_json TEXT NOT NULL,
                status TEXT NOT NULL,
                event_id TEXT NOT NULL,
                PRIMARY KEY(arena_id, target_id)
            ) STRICT;
            CREATE TABLE IF NOT EXISTS discovery_bounty_claims (
                claim_digest TEXT PRIMARY KEY,
                proposal_digest TEXT NOT NULL UNIQUE,
                claim_json TEXT NOT NULL,
                status TEXT NOT NULL,
                event_id TEXT NOT NULL
            ) STRICT;
            CREATE TABLE IF NOT EXISTS weight_publications (
                record_digest TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL UNIQUE,
                projection_digest TEXT NOT NULL,
                projection_json TEXT NOT NULL,
                record_json TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_block INTEGER NOT NULL
            ) STRICT;
            """
        )
        schema = self._db.execute(
            "SELECT value FROM metadata WHERE key='schema'"
        ).fetchone()
        if schema is None:
            self._db.execute("INSERT INTO metadata(key,value) VALUES('schema','2')")
        elif schema["value"] == "1":
            self._db.execute("UPDATE metadata SET value='2' WHERE key='schema'")
        elif schema["value"] != "2":
            raise IntakeError("intake database schema is unsupported")

    def _bind_scope(self) -> None:
        encoded = json.dumps(self.scope.to_dict(), separators=(",", ":"), sort_keys=True)
        row = self._db.execute(
            "SELECT value FROM metadata WHERE key='intake_scope'"
        ).fetchone()
        if row is None:
            self._db.execute(
                "INSERT INTO metadata(key,value) VALUES('intake_scope',?)", (encoded,)
            )
        elif row["value"] != encoded:
            raise IntakeError("intake database belongs to another chain scope")

    def _recover_interrupted(self) -> None:
        with self._transaction():
            self._db.execute(
                "UPDATE reservations SET status='held', decision='NO_DECISION', "
                "reason='controller_restart_during_' || status "
                "WHERE status IN ('fetching','qualifying')"
            )
            self._db.execute(
                "UPDATE settlement_candidates SET status='pending',lease_id='',"
                "lease_generation=lease_generation+1,lease_expires_block=0,"
                "reason='controller_restart_during_settlement' WHERE status='leased'"
            )

    def _transaction(self):
        store = self

        class Transaction:
            def __enter__(self):
                store._db.execute("BEGIN IMMEDIATE")
                return store._db

            def __exit__(self, exc_type, _exc, _tb):
                store._db.execute("ROLLBACK" if exc_type else "COMMIT")

        return Transaction()

    def _cursor(self) -> tuple[int, str] | None:
        row = self._db.execute(
            "SELECT value FROM metadata WHERE key='finalized_cursor'"
        ).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(row["value"])
        except (TypeError, ValueError) as exc:
            raise IntakeError(f"finalized cursor is corrupt: {exc}") from None
        if (
            type(value) is not list
            or len(value) != 2
            or type(value[0]) is not int
            or value[0] < 0
            or not isinstance(value[1], str)
            or _BLOCK_HASH.fullmatch(value[1]) is None
        ):
            raise IntakeError("finalized cursor is malformed")
        return value[0], value[1]

    def finalized_cursor(self) -> tuple[int, str] | None:
        """Return the last atomically reserved finalized head, if any."""

        return self._cursor()

    def reserve_finalized(
        self,
        arrivals: Iterable[FinalizedArrival],
        *,
        finalized_block: int,
        finalized_block_hash: str,
    ) -> tuple[IntakeReservation, ...]:
        rows = tuple(arrivals)
        if any(type(row) is not FinalizedArrival for row in rows):
            raise IntakeError("finalized reservation input is not typed")
        if tuple(row.arrival_key for row in rows) != tuple(
            sorted({row.arrival_key for row in rows})
        ):
            raise IntakeError("finalized arrivals are duplicated or out of order")
        if type(finalized_block) is not int or finalized_block < 0:
            raise IntakeError("finalized block is malformed")
        if _BLOCK_HASH.fullmatch(finalized_block_hash or "") is None:
            raise IntakeError("finalized block hash is malformed")
        if any(row.block > finalized_block for row in rows):
            raise IntakeError("unfinalized arrival reached durable intake")

        inserted: list[str] = []
        with self._transaction():
            cursor = self._cursor()
            if cursor is not None and (
                finalized_block < cursor[0]
                or (finalized_block == cursor[0] and finalized_block_hash != cursor[1])
            ):
                raise IntakeError("finalized cursor regressed or changed hash")
            pending = self._db.execute(
                "SELECT COUNT(*) AS n FROM reservations WHERE status IN "
                "('reserved','fetching','transport_retry','published','qualifying','held','no_decision')"
            ).fetchone()["n"]
            for arrival in rows:
                existing = self._db.execute(
                    "SELECT * FROM reservations WHERE reservation_id=?",
                    (arrival.reservation_id,),
                ).fetchone()
                if existing is not None:
                    if self._row(existing).arrival != arrival:
                        raise IntakeError("reservation ID collision changed arrival bytes")
                    continue
                epoch = arrival.block // self.policy.epoch_blocks
                if arrival.block % self.policy.epoch_blocks >= (
                    self.policy.epoch_blocks - self.policy.cutoff_blocks
                ):
                    epoch += 1
                hotkey_count = self._db.execute(
                    "SELECT COUNT(*) AS n FROM reservations WHERE admission_epoch=? AND hotkey=?",
                    (epoch, arrival.hotkey),
                ).fetchone()["n"]
                status, reason = "reserved", ""
                if not arrival.valid:
                    status, reason = "failed", arrival.invalid_reason
                elif hotkey_count >= self.policy.max_per_hotkey_epoch:
                    status, reason = "failed", "hotkey_epoch_admission_limit"
                elif pending >= self.policy.max_pending:
                    status, reason = "failed", "pending_queue_admission_limit"
                else:
                    pending += 1
                self._db.execute(
                    "INSERT INTO reservations(reservation_id,block,block_hash,event_index,event_subindex,"
                    "hotkey,content_hash,url,payload_digest,invalid_reason,admission_epoch,status,reason) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        arrival.reservation_id,
                        arrival.block,
                        arrival.block_hash,
                        arrival.event_index,
                        arrival.event_subindex,
                        arrival.hotkey,
                        arrival.content_hash,
                        arrival.url,
                        arrival.payload_digest,
                        arrival.invalid_reason,
                        epoch,
                        status,
                        reason,
                    ),
                )
                inserted.append(arrival.reservation_id)
            cursor_value = json.dumps(
                [finalized_block, finalized_block_hash], separators=(",", ":")
            )
            self._db.execute(
                "INSERT INTO metadata(key,value) VALUES('finalized_cursor',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (cursor_value,),
            )
        return tuple(self.get(value) for value in inserted)

    def _row(self, row: sqlite3.Row) -> IntakeReservation:
        try:
            members = tuple(json.loads(row["target_members_json"]))
            fingerprint = (
                SubmittedDeltaFingerprint.from_dict(
                    json.loads(row["delta_fingerprint_json"])
                )
                if row["delta_fingerprint_json"]
                else None
            )
        except (TypeError, ValueError) as exc:
            raise IntakeError(f"reservation provenance is corrupt: {exc}") from None
        arrival = FinalizedArrival(
            row["hotkey"], row["content_hash"], row["url"], row["block"],
            row["block_hash"], row["event_index"], row["event_subindex"],
            row["payload_digest"], row["invalid_reason"],
        )
        return IntakeReservation(
            row["reservation_id"], arrival, row["admission_epoch"], row["status"],
            row["target_id"], members, fingerprint, row["transport_attempts"],
            row["publication_digest"], row["publication_root"],
            row["qualification_authority_digest"], row["qualification_evidence_digest"],
            row["decision"], row["reason"],
        )

    def get(self, reservation_id: str) -> IntakeReservation:
        row = self._db.execute(
            "SELECT * FROM reservations WHERE reservation_id=?", (reservation_id,)
        ).fetchone()
        if row is None:
            raise IntakeError("unknown intake reservation")
        return self._row(row)

    def all(self) -> tuple[IntakeReservation, ...]:
        return tuple(self._row(row) for row in self._db.execute(
            "SELECT * FROM reservations ORDER BY block,event_index,event_subindex,hotkey,content_hash"
        ))

    def pending(self, *, limit: int | None = None) -> tuple[IntakeReservation, ...]:
        bound = self.policy.max_cohort if limit is None else limit
        if type(bound) is not int or bound <= 0 or bound > self.policy.max_pending:
            raise IntakeError("pending reservation limit is invalid")
        rows = self._db.execute(
            "SELECT * FROM reservations WHERE status IN ('reserved','transport_retry') "
            "AND transport_attempts < ? ORDER BY block,event_index,event_subindex,hotkey,content_hash LIMIT ?",
            (self.policy.max_transport_retries, bound),
        )
        return tuple(self._row(row) for row in rows)

    def mark_fetching(self, reservation_id: str) -> IntakeReservation:
        with self._transaction():
            row = self.get(reservation_id)
            if row.status not in {"reserved", "transport_retry"}:
                raise IntakeError("only pending intake may begin transport")
            attempts = row.transport_attempts + 1
            status = "fetching" if attempts <= self.policy.max_transport_retries else "held"
            reason = "" if status == "fetching" else "transport_retry_limit"
            self._db.execute(
                "UPDATE reservations SET status=?,transport_attempts=?,reason=? WHERE reservation_id=?",
                (status, attempts, reason, reservation_id),
            )
        return self.get(reservation_id)

    def mark_transport_retry(self, reservation_id: str, reason: str) -> IntakeReservation:
        row = self.get(reservation_id)
        exhausted = row.transport_attempts >= self.policy.max_transport_retries
        return self._transition(
            reservation_id,
            {"fetching"},
            "held" if exhausted else "transport_retry",
            "NO_DECISION",
            "transport_retry_limit" if exhausted else reason,
        )

    def mark_failed(self, reservation_id: str, reason: str) -> IntakeReservation:
        return self._transition(
            reservation_id, {"fetching", "published"}, "failed", "FAIL", reason
        )

    def mark_held(self, reservation_id: str, reason: str) -> IntakeReservation:
        return self._transition(
            reservation_id,
            {"reserved", "fetching", "transport_retry", "published", "qualifying", "no_decision"},
            "held",
            "NO_DECISION",
            reason,
        )

    def mark_published(
        self,
        reservation_id: str,
        *,
        delta_fingerprint: SubmittedDeltaFingerprint,
        publication_digest: str,
        publication_root: str | Path,
    ) -> IntakeReservation:
        if type(delta_fingerprint) is not SubmittedDeltaFingerprint:
            raise IntakeError("publication requires a typed submitted-delta fingerprint")
        target_id = delta_fingerprint.target_id
        members = delta_fingerprint.members
        require_sha256_hex(publication_digest, field="publication_digest")
        with self._transaction():
            row = self.get(reservation_id)
            if row.status != "fetching":
                raise IntakeError("publication requires an active transport")
            count = self._db.execute(
                "SELECT COUNT(*) AS n FROM reservations WHERE admission_epoch=? AND target_id=?",
                (row.admission_epoch, target_id),
            ).fetchone()["n"]
            status, reason = "published", ""
            if count >= self.policy.max_per_target_epoch:
                status, reason = "failed", "target_epoch_admission_limit"
            self._db.execute(
                "UPDATE reservations SET status=?,target_id=?,target_members_json=?,delta_fingerprint_json=?,"
                "publication_digest=?,publication_root=?,decision='',reason=? WHERE reservation_id=?",
                (
                    status, target_id, json.dumps(members, separators=(",", ":")),
                    json.dumps(delta_fingerprint.to_dict(), separators=(",", ":"), sort_keys=True),
                    publication_digest, str(publication_root), reason, reservation_id,
                ),
            )
        return self.get(reservation_id)

    def published(self, *, limit: int | None = None) -> tuple[IntakeReservation, ...]:
        bound = self.policy.max_cohort if limit is None else limit
        if type(bound) is not int or bound <= 0 or bound > self.policy.max_cohort:
            raise IntakeError("published cohort limit is invalid")
        first = self._db.execute(
            "SELECT retry_group_digest FROM reservations WHERE status='published' "
            "ORDER BY block,event_index,event_subindex,hotkey,content_hash LIMIT 1"
        ).fetchone()
        if first is not None and first["retry_group_digest"]:
            rows = self._db.execute(
                "SELECT * FROM reservations WHERE status='published' "
                "AND retry_group_digest=? ORDER BY retry_position LIMIT ?",
                (first["retry_group_digest"], bound),
            )
        else:
            rows = self._db.execute(
                "SELECT * FROM reservations WHERE status='published' "
                "AND retry_group_digest='' "
                "ORDER BY block,event_index,event_subindex,hotkey,content_hash LIMIT ?",
                (bound,),
            )
        return tuple(self._row(row) for row in rows)

    def settlement_blockers(self, reservation_id: str) -> tuple[IntakeReservation, ...]:
        candidate = self.get(reservation_id)
        if not candidate.target_members:
            raise IntakeError("candidate has no resolved target members")
        blockers: list[IntakeReservation] = []
        for row in self.all():
            if row.arrival.arrival_key >= candidate.arrival.arrival_key:
                break
            if row.status in _TERMINAL:
                continue
            if not row.target_members or set(row.target_members) & set(candidate.target_members):
                blockers.append(row)
        return tuple(blockers)

    def copy_predecessors(self, reservation_id: str) -> tuple[IntakeReservation, ...]:
        candidate = self.get(reservation_id)
        if candidate.delta_fingerprint is None:
            raise IntakeError("candidate has no submitted-delta fingerprint")
        matches: list[IntakeReservation] = []
        for row in self.all():
            if row.arrival.arrival_key >= candidate.arrival.arrival_key:
                break
            if (
                row.arrival.hotkey == candidate.arrival.hotkey
                or row.delta_fingerprint is None
            ):
                continue
            if compare_submitted_deltas(
                row.delta_fingerprint, candidate.delta_fingerprint
            ).authoritative:
                matches.append(row)
        return tuple(matches)

    def reconcile_copies(self) -> tuple[tuple[str, str], ...]:
        """Idempotently demote every unresolved later copy in finalized order."""

        dispositions = []
        for row in self.all():
            if row.delta_fingerprint is None or row.status in {"failed", "expired"}:
                continue
            predecessors = self.copy_predecessors(row.reservation_id)
            if predecessors:
                predecessor = predecessors[0]
                self.mark_copy(row.reservation_id, predecessor.reservation_id)
                dispositions.append((row.reservation_id, predecessor.reservation_id))
        return tuple(dispositions)

    def mark_copy(self, reservation_id: str, predecessor_id: str) -> IntakeReservation:
        predecessor = self.get(predecessor_id)
        candidate = self.get(reservation_id)
        if predecessor.arrival.arrival_key >= candidate.arrival.arrival_key:
            raise IntakeError("copy predecessor is not earlier in finalized order")
        if predecessor not in self.copy_predecessors(reservation_id):
            raise IntakeError("claimed predecessor is not an authoritative delta copy")
        return self._transition(
            reservation_id,
            {"published", "qualifying", "qualified", "held", "no_decision"},
            "failed",
            "FAIL",
            f"copy_of:{predecessor.reservation_id}",
        )

    def mark_qualifying(
        self,
        reservation_id: str,
        authority_digest: str,
        authority_manifest: dict[str, object],
    ) -> IntakeReservation:
        require_sha256_hex(authority_digest, field="qualification_authority_digest")
        if type(authority_manifest) is not dict or not authority_manifest:
            raise IntakeError("qualification authority manifest is not a closed object")
        authority_json = json.dumps(
            authority_manifest, separators=(",", ":"), sort_keys=True
        )
        if len(authority_json.encode("utf-8")) > 1 << 20:
            raise IntakeError("qualification authority manifest is oversized")
        with self._transaction():
            row = self.get(reservation_id)
            if row.status != "published":
                raise IntakeError("only published intake may enter qualification")
            self._db.execute(
                "UPDATE reservations SET status='qualifying',qualification_authority_digest=?,"
                "qualification_authority_json=?,"
                "decision='',reason='' WHERE reservation_id=?",
                (authority_digest, authority_json, reservation_id),
            )
        return self.get(reservation_id)

    def mark_outcome(
        self,
        reservation_id: str,
        *,
        decision: str,
        attempt_ref: EvidenceArtifactRef | None = None,
        report_digest: str = "",
        failure_digest: str = "",
        reason: str = "",
    ) -> IntakeReservation:
        if decision not in {"PASS", "FAIL", "NO_DECISION"}:
            raise IntakeError("qualification decision is unsupported")
        report_based = attempt_ref is not None or bool(report_digest)
        if report_based and (
            type(attempt_ref) is not EvidenceArtifactRef or not report_digest
        ):
            raise IntakeError("qualification report evidence is incomplete")
        if report_digest:
            require_sha256_hex(report_digest, field="qualification report digest")
        if failure_digest:
            require_sha256_hex(failure_digest, field="qualification failure digest")
        if decision == "NO_DECISION":
            if report_based == bool(failure_digest):
                raise IntakeError("NO_DECISION requires one report or failure product")
        elif not report_based or failure_digest:
            raise IntakeError("PASS/FAIL requires a retained attempt and report")
        evidence_digest = attempt_ref.sha256 if attempt_ref is not None else failure_digest
        attempt_json = (
            json.dumps(attempt_ref.to_dict(), separators=(",", ":"), sort_keys=True)
            if attempt_ref is not None
            else ""
        )
        status = {"PASS": "qualified", "FAIL": "failed", "NO_DECISION": "no_decision"}[decision]
        with self._transaction():
            row = self.get(reservation_id)
            authority_json = self._db.execute(
                "SELECT qualification_authority_json FROM reservations WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()["qualification_authority_json"]
            if (
                row.status != "qualifying"
                or not row.qualification_authority_digest
                or not authority_json
            ):
                raise IntakeError("qualification outcome lacks an active authority")
            attempt = self._db.execute(
                "SELECT COUNT(*) AS n FROM qualification_dispositions WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()["n"]
            self._db.execute(
                "INSERT INTO qualification_dispositions(reservation_id,attempt_index,authority_digest,"
                "authority_manifest_json,evidence_digest,attempt_ref_json,report_digest,failure_digest,"
                "decision,reason) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    reservation_id, attempt, row.qualification_authority_digest,
                    authority_json, evidence_digest, attempt_json, report_digest,
                    failure_digest, decision, reason,
                ),
            )
            self._db.execute(
                "UPDATE reservations SET status=?,decision=?,reason=?,"
                "qualification_evidence_digest=? WHERE reservation_id=?",
                (status, decision, reason, evidence_digest, reservation_id),
            )
        return self.get(reservation_id)

    def qualification_dispositions(self, reservation_id: str) -> tuple[dict[str, object], ...]:
        self.get(reservation_id)
        result = []
        for row in self._db.execute(
            "SELECT attempt_index,authority_digest,authority_manifest_json,evidence_digest,"
            "attempt_ref_json,report_digest,failure_digest,decision,reason "
            "FROM qualification_dispositions WHERE reservation_id=? ORDER BY attempt_index",
            (reservation_id,),
        ):
            value = dict(row)
            value["authority_manifest"] = json.loads(
                value.pop("authority_manifest_json")
            )
            attempt_ref_json = value.pop("attempt_ref_json")
            value["attempt_ref"] = (
                json.loads(attempt_ref_json) if attempt_ref_json else None
            )
            result.append(value)
        return tuple(result)

    def apply_qualification_batch(
        self,
        batch,
        *,
        evidence_root: str | Path | None = None,
    ) -> tuple[IntakeReservation, ...]:
        """Persist one typed cohort result and its retry groups atomically."""

        from optima.eval.qualification_intake import QualificationIntakeBatch
        from optima.eval.qualification import QualificationDecision
        from optima.settlement import SettlementCandidate

        if type(batch) is not QualificationIntakeBatch:
            raise IntakeError("qualification batch is not exactly typed")
        if any(
            outcome.decision is QualificationDecision.PASS
            and type(outcome.settlement_candidate) is not SettlementCandidate
            for outcome in batch.outcomes
        ):
            raise IntakeError("PASS qualification lacks a settlement projection")
        root = None if evidence_root is None else Path(evidence_root)
        if any(
            outcome.decision is QualificationDecision.PASS
            for outcome in batch.outcomes
        ) and (
            root is None
            or not root.is_absolute()
            or root != Path(os.path.normpath(root))
        ):
            raise IntakeError("PASS qualification lacks a canonical evidence root")
        retry: dict[str, tuple[str, int, str]] = {}
        if batch.retry_plan is not None:
            for group_index, group in enumerate(batch.retry_plan.reservation_groups):
                group_digest = canonical_digest(
                    "optima.chain.qualification-retry-group",
                    {
                        "authority_manifest_digest": batch.authority_manifest_digest,
                        "group_index": group_index,
                        "members": list(group),
                        "strategy": batch.retry_plan.strategy,
                    },
                )
                for position, reservation_id in enumerate(group):
                    retry[reservation_id] = (
                        group_digest,
                        position,
                        f"qualification_{batch.retry_plan.strategy}",
                    )
        with self._transaction():
            for outcome in batch.outcomes:
                reservation_id = outcome.reservation_digest
                row = self.get(reservation_id)
                if (
                    row.status != "qualifying"
                    or row.qualification_authority_digest
                    != batch.authority_manifest_digest
                    or row.delta_fingerprint is None
                    or row.delta_fingerprint.selected_delta_digest
                    != outcome.selected_delta_digest
                ):
                    raise IntakeError("qualification batch differs from active authority")
                authority_json = self._db.execute(
                    "SELECT qualification_authority_json FROM reservations WHERE reservation_id=?",
                    (reservation_id,),
                ).fetchone()["qualification_authority_json"]
                attempt_ref = (
                    batch.attempt_ref
                    if outcome.attempt_artifact_sha256 is not None
                    else None
                )
                attempt_json = (
                    json.dumps(
                        attempt_ref.to_dict(), separators=(",", ":"), sort_keys=True
                    )
                    if attempt_ref is not None
                    else ""
                )
                evidence = (
                    attempt_ref.sha256
                    if attempt_ref is not None
                    else outcome.failure_digest or ""
                )
                attempt = self._db.execute(
                    "SELECT COUNT(*) AS n FROM qualification_dispositions WHERE reservation_id=?",
                    (reservation_id,),
                ).fetchone()["n"]
                self._db.execute(
                    "INSERT INTO qualification_dispositions(reservation_id,attempt_index,"
                    "authority_digest,authority_manifest_json,evidence_digest,attempt_ref_json,"
                    "report_digest,failure_digest,decision,reason) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        reservation_id, attempt, batch.authority_manifest_digest,
                        authority_json, evidence, attempt_json,
                        outcome.report_digest or "", outcome.failure_digest or "",
                        outcome.decision.value, outcome.reason,
                    ),
                )
                candidate = outcome.settlement_candidate
                if candidate is not None:
                    if (
                        type(candidate) is not SettlementCandidate
                        or candidate.reservation_digest != reservation_id
                        or candidate.hotkey != row.arrival.hotkey
                        or (
                            candidate.finalized_block,
                            candidate.event_index,
                            candidate.event_subindex,
                        )
                        != (
                            row.arrival.block,
                            row.arrival.event_index,
                            row.arrival.event_subindex,
                        )
                        or candidate.target_id != row.target_id
                        or candidate.members != row.target_members
                        or candidate.selected_delta_digest
                        != row.delta_fingerprint.selected_delta_digest
                        or candidate.qualification_authority_digest
                        != batch.authority_manifest_digest
                        or candidate.qualification_attempt_digest != evidence
                        or candidate.qualification_report_digest
                        != outcome.report_digest
                    ):
                        raise IntakeError(
                            "settlement candidate differs from retained qualification"
                        )
                    self.evaluation_stack(candidate.arena_digest)
                    candidate_json = json.dumps(
                        candidate.to_dict(), separators=(",", ":"), sort_keys=True
                    )
                    self._db.execute(
                        "INSERT INTO settlement_candidates(reservation_id,authority_digest,"
                        "candidate_digest,candidate_json,evidence_root,status) "
                        "VALUES(?,?,?,?,?, 'pending')",
                        (
                            reservation_id,
                            batch.authority_manifest_digest,
                            candidate.digest,
                            candidate_json,
                            str(root),
                        ),
                    )
                if reservation_id in retry:
                    group, position, reason = retry[reservation_id]
                    status = (
                        "published"
                        if attempt + 1 < self.policy.max_qualification_retries
                        else "held"
                    )
                    self._db.execute(
                        "UPDATE reservations SET status=?,decision=?,reason=?,"
                        "retry_group_digest=?,retry_position=?,qualification_authority_digest='',"
                        "qualification_authority_json='',qualification_evidence_digest='' "
                        "WHERE reservation_id=?",
                        (
                            status, "" if status == "published" else "NO_DECISION",
                            reason, group, position, reservation_id,
                        ),
                    )
                else:
                    status = {
                        "PASS": "qualified", "FAIL": "failed"
                    }[outcome.decision.value]
                    self._db.execute(
                        "UPDATE reservations SET status=?,decision=?,reason=?,"
                        "qualification_evidence_digest=?,retry_group_digest='',retry_position=0 "
                        "WHERE reservation_id=?",
                        (
                            status, outcome.decision.value, outcome.reason,
                            evidence, reservation_id,
                        ),
                    )
        return tuple(self.get(row.reservation_digest) for row in batch.outcomes)

    # ---- transactional settlement and evaluation-stack authority ----

    def _initialize_evaluation_stack_row(
        self,
        manifest: EvaluationStackManifest,
        *,
        tree_digest: str,
    ) -> None:
        """Initialize one stack inside the caller's transaction, idempotently."""

        from optima.stack_manifest import EvaluationStackManifest

        if type(manifest) is not EvaluationStackManifest:
            raise IntakeError("initial evaluation stack is not exactly typed")
        require_sha256_hex(tree_digest, field="tree_digest")
        arena = manifest.arena_digest
        genesis = canonical_digest(
            "optima.chain.evaluation-stack-genesis",
            {
                "arena_digest": arena,
                "stack_digest": manifest.digest,
                "tree_digest": tree_digest,
            },
        )
        encoded = json.dumps(
            manifest.to_dict(), separators=(",", ":"), sort_keys=True
        )
        existing = self._db.execute(
            "SELECT * FROM evaluation_stacks WHERE arena_id=?", (arena,)
        ).fetchone()
        if existing is None:
            self._db.execute(
                "INSERT INTO evaluation_stacks(arena_id,generation,stack_digest,"
                "tree_digest,stack_json,transition_event_id) VALUES(?,0,?,?,?,?)",
                (arena, manifest.digest, tree_digest, encoded, genesis),
            )
        elif existing["generation"] == 0 and (
            existing["stack_digest"] != manifest.digest
            or existing["tree_digest"] != tree_digest
            or existing["stack_json"] != encoded
        ):
            raise IntakeError("genesis qualification names another incumbent")

    def initialize_evaluation_stack(
        self,
        manifest: EvaluationStackManifest,
        *,
        tree_digest: str,
    ) -> EvaluationStackState:
        """Install one exact genesis incumbent, or reopen the identical state."""

        with self._transaction():
            self._initialize_evaluation_stack_row(manifest, tree_digest=tree_digest)
        state = self.evaluation_stack(manifest.arena_digest)
        if (
            state.manifest.digest != manifest.digest
            or state.tree_digest != tree_digest
        ):
            raise IntakeError("evaluation stack is already initialized differently")
        return state

    def evaluation_stack(self, arena_digest: str) -> EvaluationStackState:
        from optima.stack_manifest import EvaluationStackManifest

        require_sha256_hex(arena_digest, field="arena_digest")
        row = self._db.execute(
            "SELECT * FROM evaluation_stacks WHERE arena_id=?", (arena_digest,)
        ).fetchone()
        if row is None:
            raise IntakeError("evaluation stack is not initialized")
        try:
            manifest = EvaluationStackManifest.from_dict(json.loads(row["stack_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IntakeError(f"evaluation stack state is corrupt: {exc}") from None
        if manifest.digest != row["stack_digest"]:
            raise IntakeError("evaluation stack digest differs from stored bytes")
        return EvaluationStackState(
            row["arena_id"], row["generation"], manifest, row["tree_digest"],
            row["transition_event_id"],
        )

    @staticmethod
    def _settlement_candidate(row: sqlite3.Row) -> SettlementCandidate:
        from optima.settlement import SettlementCandidate

        try:
            candidate = SettlementCandidate.from_dict(json.loads(row["candidate_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IntakeError(f"settlement candidate is corrupt: {exc}") from None
        if candidate.digest != row["candidate_digest"]:
            raise IntakeError("settlement candidate digest differs from stored bytes")
        return candidate

    def _event_head(self) -> tuple[int, str]:
        row = self._db.execute(
            "SELECT sequence,event_digest FROM settlement_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        return (0, "") if row is None else (row["sequence"] + 1, row["event_digest"])

    def _settlement_evidence_metadata(
        self,
        candidate: SettlementCandidate,
    ):
        from optima.eval.evidence_store import EvidenceArtifactRef
        from optima.settlement import SettlementEvidence

        row = self._db.execute(
            "SELECT sc.evidence_root,sc.candidate_digest,r.status,r.decision,"
            "qd.authority_digest,qd.attempt_ref_json,qd.report_digest,qd.decision AS qdecision "
            "FROM settlement_candidates sc JOIN reservations r USING(reservation_id) "
            "JOIN qualification_dispositions qd USING(reservation_id) "
            "WHERE sc.reservation_id=? AND qd.evidence_digest=? "
            "ORDER BY qd.attempt_index DESC LIMIT 1",
            (candidate.reservation_digest, candidate.qualification_attempt_digest),
        ).fetchone()
        if (
            row is None
            or row["candidate_digest"] != candidate.digest
            or row["status"] != "qualified"
            or row["decision"] != "PASS"
            or row["qdecision"] != "PASS"
            or row["authority_digest"] != candidate.qualification_authority_digest
            or row["report_digest"] != candidate.qualification_report_digest
            or not row["attempt_ref_json"]
            or not row["evidence_root"]
        ):
            raise IntakeError("settlement evidence no longer has standing authority")
        try:
            reference = EvidenceArtifactRef.from_dict(
                json.loads(row["attempt_ref_json"])
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IntakeError(f"settlement evidence reference is corrupt: {exc}") from None
        if reference.sha256 != candidate.qualification_attempt_digest:
            raise IntakeError("settlement attempt reference differs from candidate")
        return (
            Path(row["evidence_root"]),
            SettlementEvidence(
                candidate.digest,
                candidate.reservation_digest,
                candidate.qualification_authority_digest,
                reference,
                candidate.qualification_report_digest,
            ),
        )

    def reopen_settlement_evidence(
        self,
        candidate: SettlementCandidate,
    ):
        """Reopen retained qualification bytes without duplicating their grader."""

        from optima.eval.evidence_store import EvidenceStoreError, reopen_evidence
        from optima.settlement import SettlementCandidate

        if type(candidate) is not SettlementCandidate:
            raise IntakeError("settlement evidence candidate is not exactly typed")
        root, receipt = self._settlement_evidence_metadata(candidate)
        try:
            reopen_evidence(root, receipt.attempt_ref)
        except (EvidenceStoreError, OSError) as exc:
            raise IntakeError(f"retained settlement evidence cannot reopen: {exc}") from None
        return receipt

    def _economic_blockers(
        self,
        candidate: SettlementCandidate,
        *,
        cohort_ids: frozenset[str],
    ) -> tuple[str, ...]:
        blockers: list[str] = []
        for row in self.all():
            if row.arrival.arrival_key >= self.get(
                candidate.reservation_digest
            ).arrival.arrival_key:
                break
            if row.reservation_id in cohort_ids:
                continue
            if row.status in {"failed", "expired"}:
                continue
            if row.target_members and not (
                set(row.target_members) & set(candidate.members)
            ):
                continue
            if row.status == "qualified":
                economic = self._db.execute(
                    "SELECT status FROM settlement_candidates WHERE reservation_id=?",
                    (row.reservation_id,),
                ).fetchone()
                if economic is not None and economic["status"] in {
                    "crowned", "neutralized", "held", "discovery_bounty"
                }:
                    continue
            blockers.append(row.reservation_id)
        return tuple(blockers)

    def lease_settlement_cohort(
        self,
        *,
        current_block: int,
        lease_blocks: int = 30,
    ) -> SettlementLease | None:
        """Lease the oldest economically unblocked retained PASS cohort."""

        if (
            type(current_block) is not int
            or current_block < 0
            or type(lease_blocks) is not int
            or lease_blocks <= 0
        ):
            raise IntakeError("settlement lease bounds are malformed")
        with self._transaction():
            self._db.execute(
                "UPDATE settlement_candidates SET status='held',lease_id='',"
                "lease_expires_block=0,reason='intake_no_longer_qualified' "
                "WHERE status IN ('pending','leased') AND reservation_id IN "
                "(SELECT reservation_id FROM reservations WHERE status!='qualified')"
            )
            self._db.execute(
                "UPDATE settlement_candidates SET status='pending',lease_id='',"
                "lease_generation=lease_generation+1,lease_expires_block=0,"
                "reason='settlement_lease_expired' WHERE status='leased' "
                "AND lease_expires_block<=?",
                (current_block,),
            )
            pending = tuple(
                self._db.execute(
                    "SELECT sc.*,r.block,r.event_index,r.event_subindex,r.hotkey,r.content_hash "
                    "FROM settlement_candidates sc JOIN reservations r USING(reservation_id) "
                    "WHERE sc.status='pending' AND r.status='qualified' "
                    "ORDER BY r.block,r.event_index,"
                    "r.event_subindex,r.hotkey,r.content_hash"
                )
            )
            chosen: tuple[sqlite3.Row, ...] | None = None
            for first in pending:
                group = tuple(
                    row for row in pending
                    if row["authority_digest"] == first["authority_digest"]
                )
                candidates = tuple(self._settlement_candidate(row) for row in group)
                if len({row.arena_digest for row in candidates}) != 1:
                    raise IntakeError("one qualification authority spans multiple arenas")
                ids = frozenset(row.reservation_digest for row in candidates)
                if any(
                    self._economic_blockers(row, cohort_ids=ids)
                    for row in candidates
                ):
                    continue
                chosen = group
                break
            if chosen is None:
                return None
            candidates = tuple(self._settlement_candidate(row) for row in chosen)
            stack = self.evaluation_stack(candidates[0].arena_digest)
            generation = max(row["lease_generation"] for row in chosen) + 1
            expires = current_block + lease_blocks
            lease_id = canonical_digest(
                "optima.chain.settlement-lease",
                {
                    "authority_digest": chosen[0]["authority_digest"],
                    "candidates": [row.digest for row in candidates],
                    "generation": generation,
                    "incumbent_generation": stack.generation,
                    "lease_block": current_block,
                },
            )
            ids = tuple(row.reservation_digest for row in candidates)
            marks = ",".join("?" for _ in ids)
            cursor = self._db.execute(
                f"UPDATE settlement_candidates SET status='leased',lease_id=?,"
                f"lease_generation=?,lease_expires_block=?,reason='' "
                f"WHERE status='pending' AND reservation_id IN ({marks})",
                (lease_id, generation, expires, *ids),
            )
            if cursor.rowcount != len(ids):
                raise IntakeError("settlement cohort changed while leasing")
            sequence, previous = self._event_head()
        return SettlementLease(
            lease_id,
            chosen[0]["authority_digest"],
            generation,
            expires,
            stack,
            candidates,
            sequence,
            previous,
        )

    def commit_settlement(
        self,
        lease: SettlementLease,
        plan: SettlementPlan,
        evidence,
        *,
        current_block: int,
    ) -> EvaluationStackState:
        """Atomically commit one independently planned retained-evidence disposition."""

        from optima.economics import DiscoveryBountyClaim, StandingRewardClaim, WEIGHT_PPM
        from optima.settlement import (
            SettlementEvidence,
            SettlementEventType,
            SettlementPlan,
            plan_settlement,
        )

        if type(lease) is not SettlementLease or type(plan) is not SettlementPlan:
            raise IntakeError("settlement commit is not exactly typed")
        receipts = tuple(evidence)
        if (
            type(current_block) is not int
            or current_block < 0
            or current_block >= lease.expires_block
            or len(receipts) != len(lease.candidates)
            or any(type(row) is not SettlementEvidence for row in receipts)
            or {row.candidate_digest for row in receipts}
            != {row.digest for row in lease.candidates}
        ):
            raise IntakeError("settlement evidence or lease deadline is invalid")
        expected = plan_settlement(
            lease.candidates,
            current_manifest=lease.stack.manifest,
            current_tree_digest=lease.stack.tree_digest,
            initial_event_sequence=lease.initial_event_sequence,
            previous_event_digest=lease.previous_event_digest,
        )
        if expected.to_dict() != plan.to_dict():
            raise IntakeError("settlement plan differs from its leased authority")
        by_digest = {row.digest: row for row in lease.candidates}
        evidence_by_candidate = {row.candidate_digest: row for row in receipts}
        with self._transaction():
            current = self.evaluation_stack(lease.stack.arena_digest)
            if current != lease.stack or self._event_head() != (
                lease.initial_event_sequence, lease.previous_event_digest
            ):
                raise IntakeError("settlement incumbent or journal advanced")
            ids = tuple(row.reservation_digest for row in lease.candidates)
            cohort_ids = frozenset(ids)
            if any(
                self._economic_blockers(candidate, cohort_ids=cohort_ids)
                for candidate in lease.candidates
            ):
                raise IntakeError("settlement priority changed while evidence was open")
            for candidate in lease.candidates:
                _root, expected_receipt = self._settlement_evidence_metadata(candidate)
                if expected_receipt != evidence_by_candidate[candidate.digest]:
                    raise IntakeError("settlement evidence changed after reopening")
            marks = ",".join("?" for _ in ids)
            active = tuple(
                self._db.execute(
                    f"SELECT sc.reservation_id,sc.candidate_digest FROM settlement_candidates sc "
                    f"JOIN reservations r USING(reservation_id) WHERE sc.status='leased' "
                    f"AND r.status='qualified' AND sc.lease_id=? AND sc.lease_generation=? "
                    f"AND sc.reservation_id IN ({marks})",
                    (lease.lease_id, lease.generation, *ids),
                )
            )
            if len(active) != len(ids) or {
                row["candidate_digest"] for row in active
            } != set(by_digest):
                raise IntakeError("settlement lease is stale or incomplete")

            # Retire/neutralize old families before installing the winner family.
            for event in plan.events:
                if event.event_type in {
                    SettlementEventType.RETIREMENT,
                    SettlementEventType.NEUTRALIZATION,
                }:
                    self._db.execute(
                        "UPDATE standing_reward_claims SET status='inactive',event_id=? "
                        "WHERE arena_id=? AND target_id=?",
                        (event.digest, lease.stack.arena_digest, event.target_id),
                    )

            disposition: dict[str, str] = {}
            for event in plan.events:
                candidate = by_digest[event.candidate_digest]
                event_json = json.dumps(
                    event.to_dict(), separators=(",", ":"), sort_keys=True
                )
                self._db.execute(
                    "INSERT INTO settlement_events(sequence,event_id,event_type,reservation_id,"
                    "arena_id,target_id,event_digest,event_json) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        event.sequence,
                        event.digest,
                        event.event_type.value,
                        candidate.reservation_digest,
                        candidate.arena_digest,
                        event.target_id,
                        event.digest,
                        event_json,
                    ),
                )
                if event.event_type is SettlementEventType.HOLD:
                    disposition[candidate.digest] = "held"
                elif event.event_type is SettlementEventType.DISCOVERY_BOUNTY:
                    speedup_ppm = int(
                        (Decimal(candidate.speedup) * WEIGHT_PPM).to_integral_value(
                            rounding=ROUND_FLOOR
                        )
                    )
                    claim = DiscoveryBountyClaim(
                        candidate.proposal_digest,
                        evidence_by_candidate[candidate.digest].digest,
                        candidate.hotkey,
                        max(1, speedup_ppm - WEIGHT_PPM),
                        candidate.finalized_block,
                    )
                    self._db.execute(
                        "INSERT INTO discovery_bounty_claims(claim_digest,proposal_digest,"
                        "claim_json,status,event_id) VALUES(?,?,?,?,?)",
                        (
                            claim.digest,
                            claim.proposal_digest,
                            json.dumps(claim.to_dict(), separators=(",", ":"), sort_keys=True),
                            "active",
                            event.digest,
                        ),
                    )
                    disposition[candidate.digest] = "discovery_bounty"
                elif event.event_type is SettlementEventType.CROWN:
                    assert candidate.candidate_manifest is not None
                    contribution = candidate.candidate_manifest.entries[candidate.target_id]
                    speedup_ppm = int(
                        (Decimal(candidate.speedup) * WEIGHT_PPM).to_integral_value(
                            rounding=ROUND_FLOOR
                        )
                    )
                    claim = StandingRewardClaim(
                        candidate.arena_digest,
                        candidate.target_id,
                        contribution.target_spec_digest,
                        contribution.digest,
                        candidate.hotkey,
                        speedup_ppm,
                        candidate.finalized_block,
                        evidence_by_candidate[candidate.digest].digest,
                    )
                    self._db.execute(
                        "INSERT INTO standing_reward_claims(arena_id,target_id,claim_digest,"
                        "claim_json,status,event_id) VALUES(?,?,?,?, 'active',?) "
                        "ON CONFLICT(arena_id,target_id) DO UPDATE SET "
                        "claim_digest=excluded.claim_digest,claim_json=excluded.claim_json,"
                        "status='active',event_id=excluded.event_id",
                        (
                            candidate.arena_digest,
                            candidate.target_id,
                            claim.digest,
                            json.dumps(claim.to_dict(), separators=(",", ":"), sort_keys=True),
                            event.digest,
                        ),
                    )
                    disposition[candidate.digest] = "crowned"

            if set(disposition) != set(by_digest):
                raise IntakeError("settlement plan did not dispose every leased candidate")
            for digest, status in disposition.items():
                candidate = by_digest[digest]
                self._db.execute(
                    "UPDATE settlement_candidates SET status=?,lease_id='',"
                    "lease_expires_block=0,reason=?,settlement_evidence_digest=? "
                    "WHERE reservation_id=?",
                    (
                        status,
                        status,
                        evidence_by_candidate[digest].digest,
                        candidate.reservation_digest,
                    ),
                )

            if plan.transition is not None:
                manifest = plan.transition.manifest
                encoded = json.dumps(
                    manifest.to_dict(), separators=(",", ":"), sort_keys=True
                )
                transition_id = plan.events[-1].digest
                cursor = self._db.execute(
                    "UPDATE evaluation_stacks SET generation=generation+1,stack_digest=?,"
                    "tree_digest=?,stack_json=?,transition_event_id=? WHERE arena_id=? "
                    "AND generation=? AND stack_digest=? AND tree_digest=?",
                    (
                        manifest.digest,
                        plan.transition.after.tree_digest,
                        encoded,
                        transition_id,
                        lease.stack.arena_digest,
                        lease.stack.generation,
                        lease.stack.manifest.digest,
                        lease.stack.tree_digest,
                    ),
                )
                if cursor.rowcount != 1:
                    raise IntakeError("evaluation stack changed during settlement commit")
        return self.evaluation_stack(lease.stack.arena_digest)

    def active_reward_claims(self) -> tuple[tuple[object, ...], tuple[object, ...]]:
        """Reopen all active standing and discovery claims, or fail as one unit."""

        from optima.economics import DiscoveryBountyClaim, StandingRewardClaim

        standing = []
        for row in self._db.execute(
            "SELECT claim_digest,claim_json FROM standing_reward_claims "
            "WHERE status='active' ORDER BY arena_id,target_id"
        ):
            try:
                claim = StandingRewardClaim.from_dict(json.loads(row["claim_json"]))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise IntakeError(f"standing reward claim is corrupt: {exc}") from None
            if claim.digest != row["claim_digest"]:
                raise IntakeError("standing reward claim digest differs")
            standing.append(claim)
        discovery = []
        for row in self._db.execute(
            "SELECT claim_digest,claim_json FROM discovery_bounty_claims "
            "WHERE status='active' ORDER BY proposal_digest"
        ):
            try:
                claim = DiscoveryBountyClaim.from_dict(json.loads(row["claim_json"]))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise IntakeError(f"discovery reward claim is corrupt: {exc}") from None
            if claim.digest != row["claim_digest"]:
                raise IntakeError("discovery reward claim digest differs")
            discovery.append(claim)
        return tuple(standing), tuple(discovery)

    def _reopen_claim_evidence(self, retained_digest: str, status: str):
        require_sha256_hex(retained_digest, field="retained_evidence_digest")
        row = self._db.execute(
            "SELECT * FROM settlement_candidates WHERE settlement_evidence_digest=? "
            "AND status=?",
            (retained_digest, status),
        ).fetchone()
        if row is None:
            raise IntakeError("active reward claim has no standing settlement candidate")
        candidate = self._settlement_candidate(row)
        receipt = self.reopen_settlement_evidence(candidate)
        if receipt.digest != retained_digest:
            raise IntakeError("active reward claim differs from reopened settlement evidence")
        return receipt

    def _bind_emissions_policy(self, policy_digest: str) -> None:
        require_sha256_hex(policy_digest, field="policy_digest")
        with self._transaction():
            row = self._db.execute(
                "SELECT value FROM metadata WHERE key='emissions_policy_digest'"
            ).fetchone()
            if row is None:
                self._db.execute(
                    "INSERT INTO metadata(key,value) VALUES('emissions_policy_digest',?)",
                    (policy_digest,),
                )
            elif row["value"] != policy_digest:
                raise IntakeError(
                    "emissions policy differs from the bound validator consensus state"
                )

    def build_weight_projection(
        self,
        *,
        policy,
        context,
        catalogs: Mapping[str, object],
        netuid: int,
    ) -> WeightProjection:
        """Build one global all-arena vector from the complete retained authority."""

        from optima.chain.weights import WeightProjection
        from optima.economics import (
            ArenaRewardAuthority,
            EmissionsPolicyManifest,
            GlobalRewardProjectionContext,
            project_global_rewards,
        )
        from optima.target_catalog import TargetCatalog

        if (
            type(policy) is not EmissionsPolicyManifest
            or type(context) is not GlobalRewardProjectionContext
            or not isinstance(catalogs, Mapping)
            or type(netuid) is not int
            or netuid < 0
        ):
            raise IntakeError("weight projection authority is malformed")
        standing, discovery = self.active_reward_claims()
        by_arena: dict[str, list[object]] = {}
        for claim in standing:
            by_arena.setdefault(claim.arena_digest, []).append(claim)
        states = self.evaluation_stacks()
        if set(by_arena) - {row.arena_digest for row in states}:
            raise IntakeError("active reward claim belongs to an absent evaluation arena")
        if set(catalogs) != {row.arena_digest for row in states}:
            raise IntakeError("reward catalogs do not cover every evaluation arena")
        for claim in standing:
            self._reopen_claim_evidence(claim.retained_evidence_digest, "crowned")
        for claim in discovery:
            self._reopen_claim_evidence(
                claim.retained_evidence_digest, "discovery_bounty"
            )
        authorities = []
        for state in states:
            catalog = catalogs[state.arena_digest]
            if type(catalog) is not TargetCatalog:
                raise IntakeError("reward catalog is not exactly typed")
            authorities.append(
                ArenaRewardAuthority(
                    catalog,
                    state.manifest,
                    state.generation,
                    tuple(by_arena.get(state.arena_digest, ())),
                )
            )
        projection = project_global_rewards(
            policy, context, tuple(authorities), discovery
        )
        self._bind_emissions_policy(policy.digest)
        evidence = tuple(
            sorted(
                {
                    claim.retained_evidence_digest
                    for claim in (*standing, *discovery)
                }
            )
        )
        return WeightProjection(
            context.chain_scope_digest,
            netuid,
            context.validator_hotkey,
            policy.digest,
            self.settlement_state_digest(),
            projection.digest,
            context.metagraph_digest,
            projection.arena_authority_digests,
            max((row.generation for row in states), default=0),
            context.current_block,
            len(standing),
            evidence,
            tuple(
                (row.hotkey, row.weight_ppm) for row in projection.weights
            ),
        )

    def settlement_state_digest(self) -> str:
        sequence, event = self._event_head()
        stacks = tuple(
            (row["arena_id"], row["generation"], row["stack_digest"], row["tree_digest"])
            for row in self._db.execute(
                "SELECT arena_id,generation,stack_digest,tree_digest "
                "FROM evaluation_stacks ORDER BY arena_id"
            )
        )
        candidates = tuple(
            (row["candidate_digest"], row["status"], row["lease_generation"])
            for row in self._db.execute(
                "SELECT candidate_digest,status,lease_generation FROM settlement_candidates "
                "ORDER BY reservation_id"
            )
        )
        return canonical_digest(
            "optima.chain.settlement-state",
            {
                "candidates": candidates,
                "event_head": event,
                "event_sequence": sequence,
                "stacks": stacks,
            },
        )

    def evaluation_stacks(self) -> tuple[EvaluationStackState, ...]:
        return tuple(
            self.evaluation_stack(row["arena_id"])
            for row in self._db.execute(
                "SELECT arena_id FROM evaluation_stacks ORDER BY arena_id"
            )
        )


    def requeue_qualification(
        self,
        reservation_id: str,
        *,
        reason: str,
        retry_group_digest: str,
        retry_position: int,
    ) -> IntakeReservation:
        if not reason:
            raise IntakeError("qualification requeue requires a reason")
        require_sha256_hex(retry_group_digest, field="retry_group_digest")
        if type(retry_position) is not int or retry_position < 0:
            raise IntakeError("qualification retry position is malformed")
        with self._transaction():
            row = self.get(reservation_id)
            if row.status != "no_decision":
                raise IntakeError("only a retained NO_DECISION may be requeued")
            attempts = self._db.execute(
                "SELECT COUNT(*) AS n FROM qualification_dispositions WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()["n"]
            status = "published" if attempts < self.policy.max_qualification_retries else "held"
            self._db.execute(
                "UPDATE reservations SET status=?,decision='',reason=?,"
                "retry_group_digest=?,retry_position=?,"
                "qualification_authority_digest='',qualification_authority_json='',"
                "qualification_evidence_digest='' "
                "WHERE reservation_id=?",
                (
                    status,
                    reason,
                    retry_group_digest,
                    retry_position,
                    reservation_id,
                ),
            )
        return self.get(reservation_id)

    def _transition(
        self,
        reservation_id: str,
        expected: set[str],
        status: str,
        decision: str,
        reason: str,
        *,
        evidence_digest: str = "",
    ) -> IntakeReservation:
        if status not in _STATUSES or not isinstance(reason, str) or len(reason) > 2_048:
            raise IntakeError("intake transition is malformed")
        with self._transaction():
            row = self.get(reservation_id)
            if row.status not in expected:
                raise IntakeError(f"intake transition from {row.status!r} is forbidden")
            self._db.execute(
                "UPDATE reservations SET status=?,decision=?,reason=?,"
                "qualification_evidence_digest=? WHERE reservation_id=?",
                (status, decision, reason, evidence_digest, reservation_id),
            )
        return self.get(reservation_id)

    def expire(self, reservation_id: str, *, current_block: int, reason: str) -> IntakeReservation:
        row = self.get(reservation_id)
        if type(current_block) is not int or current_block - row.arrival.block < self.policy.expiry_blocks:
            raise IntakeError("reservation is not old enough for explicit expiry")
        return self._transition(
            reservation_id,
            {"reserved", "transport_retry", "published", "held", "no_decision"},
            "expired",
            "NO_DECISION",
            reason,
        )

    def release_hold(self, reservation_id: str, *, reason: str) -> IntakeReservation:
        if not reason:
            raise IntakeError("hold release requires an operator reason")
        with self._transaction():
            row = self.get(reservation_id)
            if row.status not in {"held", "no_decision"}:
                raise IntakeError("only held intake may be released")
            status = "published" if row.publication_digest else "transport_retry"
            attempts = row.transport_attempts if row.publication_digest else 0
            self._db.execute(
                "UPDATE reservations SET status=?,decision='',reason=?,"
                "transport_attempts=?,retry_group_digest='',retry_position=0,"
                "qualification_authority_digest='',qualification_authority_json='',"
                "qualification_evidence_digest='' "
                "WHERE reservation_id=?",
                (status, reason, attempts, reservation_id),
            )
        return self.get(reservation_id)


class SQLiteWeightPublicationJournal:
    """CAS journal adapter over the same exclusive control-plane SQLite authority."""

    def __init__(self, store: FinalizedIntakeStore, projection: WeightProjection) -> None:
        from optima.chain.weights import WeightProjection

        if type(store) is not FinalizedIntakeStore or type(projection) is not WeightProjection:
            raise IntakeError("weight publication journal authority is not exactly typed")
        self.store = store
        self.projection = projection

    def _head(self) -> tuple[str, str] | None:
        row = self.store._db.execute(
            "SELECT value FROM metadata WHERE key='weight_publication_head'"
        ).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(row["value"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IntakeError(f"weight publication head is corrupt: {exc}") from None
        if type(value) is not dict or set(value) != {
            "projection_digest", "record_digest"
        }:
            raise IntakeError("weight publication head is malformed")
        require_sha256_hex(value["projection_digest"], field="projection_digest")
        require_sha256_hex(value["record_digest"], field="record_digest")
        return value["projection_digest"], value["record_digest"]

    def load(self) -> WeightPublicationRecord | None:
        from optima.chain.weights import WeightPublicationRecord

        head = self._head()
        if head is None:
            return None
        row = self.store._db.execute(
            "SELECT record_json FROM weight_publications WHERE record_digest=?",
            (head[1],),
        ).fetchone()
        if row is None:
            raise IntakeError("weight publication head has no retained record")
        try:
            record = WeightPublicationRecord.from_dict(json.loads(row["record_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IntakeError(f"weight publication record is corrupt: {exc}") from None
        if record.digest != head[1] or record.projection_digest != head[0]:
            raise IntakeError("weight publication head differs from retained record")
        seen: set[str] = set()
        current = record
        while True:
            if current.digest in seen:
                raise IntakeError("weight publication journal contains a cycle")
            seen.add(current.digest)
            prior = current.prior_record_digest
            if prior is None:
                break
            predecessor = self.store._db.execute(
                "SELECT record_json FROM weight_publications WHERE record_digest=?",
                (prior,),
            ).fetchone()
            if predecessor is None:
                raise IntakeError("weight publication predecessor is missing")
            try:
                current = WeightPublicationRecord.from_dict(
                    json.loads(predecessor["record_json"])
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise IntakeError(
                    f"weight publication predecessor is corrupt: {exc}"
                ) from None
            if current.digest != prior:
                raise IntakeError("weight publication predecessor digest differs")
        return record

    def compare_and_swap(
        self,
        expected_record_digest: str | None,
        replacement: WeightPublicationRecord,
    ) -> None:
        from optima.chain.weights import WeightPublicationRecord

        if type(replacement) is not WeightPublicationRecord:
            raise IntakeError("weight publication replacement is not exactly typed")
        if expected_record_digest is not None:
            require_sha256_hex(expected_record_digest, field="expected_record_digest")
        if replacement.prior_record_digest != expected_record_digest:
            raise IntakeError("weight publication replacement does not bind the CAS head")
        with self.store._transaction():
            head = self._head()
            observed = None if head is None else head[1]
            if observed != expected_record_digest:
                raise IntakeError("weight publication journal compare-and-swap failed")
            previous = self.store._db.execute(
                "SELECT sequence,projection_json FROM weight_publications "
                "WHERE projection_digest=? ORDER BY sequence DESC LIMIT 1",
                (replacement.projection_digest,),
            ).fetchone()
            if replacement.projection_digest == self.projection.digest:
                projection_json = json.dumps(
                    self.projection.to_dict(), separators=(",", ":"), sort_keys=True
                )
            elif previous is not None:
                projection_json = previous["projection_json"]
            else:
                raise IntakeError("publication record has no retained projection")
            sequence = self.store._db.execute(
                "SELECT COALESCE(MAX(sequence),0)+1 AS value FROM weight_publications"
            ).fetchone()["value"]
            record_json = json.dumps(
                replacement.to_dict(), separators=(",", ":"), sort_keys=True
            )
            updated_block = max(
                replacement.submit_block,
                replacement.confirmed_block,
                replacement.confirmed_last_update,
            )
            self.store._db.execute(
                "INSERT INTO weight_publications(record_digest,sequence,projection_digest,"
                "projection_json,record_json,status,updated_block) VALUES(?,?,?,?,?,?,?)",
                (
                    replacement.digest,
                    sequence,
                    replacement.projection_digest,
                    projection_json,
                    record_json,
                    replacement.status,
                    updated_block,
                ),
            )
            encoded = json.dumps(
                {
                    "projection_digest": replacement.projection_digest,
                    "record_digest": replacement.digest,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            self.store._db.execute(
                "INSERT INTO metadata(key,value) VALUES('weight_publication_head',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (encoded,),
            )

    def retained_projection(self, projection_digest: str) -> WeightProjection:
        from optima.chain.weights import WeightProjection

        require_sha256_hex(projection_digest, field="projection_digest")
        row = self.store._db.execute(
            "SELECT projection_json FROM weight_publications WHERE projection_digest=? "
            "ORDER BY sequence DESC LIMIT 1",
            (projection_digest,),
        ).fetchone()
        if row is None:
            raise IntakeError("weight projection is not retained")
        try:
            projection = WeightProjection.from_dict(json.loads(row["projection_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IntakeError(f"weight projection is corrupt: {exc}") from None
        if projection.digest != projection_digest:
            raise IntakeError("retained weight projection digest differs")
        return projection


__all__ = [
    "EvaluationStackState", "FinalizedArrival", "FinalizedIntakeStore", "IntakeError",
    "IntakePolicy", "IntakeReservation", "IntakeScope", "SQLiteWeightPublicationJournal",
    "SettlementLease",
]
