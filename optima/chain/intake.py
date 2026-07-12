"""Durable finalized-arrival authority, separate from evaluation and settlement."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import stat
import fcntl
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from optima.copy_fingerprint import (
    SubmittedDeltaFingerprint, compare_submitted_deltas,
)
from optima.eval.evidence_store import EvidenceArtifactRef
from optima.stack_identity import canonical_digest, require_sha256_hex


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
            """
        )
        schema = self._db.execute(
            "SELECT value FROM metadata WHERE key='schema'"
        ).fetchone()
        if schema is None:
            self._db.execute("INSERT INTO metadata(key,value) VALUES('schema','1')")
        elif schema["value"] != "1":
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

    def copy_successors(self, reservation_id: str) -> tuple[IntakeReservation, ...]:
        predecessor = self.get(reservation_id)
        if predecessor.delta_fingerprint is None:
            raise IntakeError("copy predecessor has no submitted-delta fingerprint")
        matches = []
        for row in self.all():
            if row.arrival.arrival_key <= predecessor.arrival.arrival_key:
                continue
            if (
                row.arrival.hotkey == predecessor.arrival.hotkey
                or row.delta_fingerprint is None
                or row.status in {"failed", "expired"}
            ):
                continue
            if compare_submitted_deltas(
                predecessor.delta_fingerprint, row.delta_fingerprint
            ).authoritative:
                matches.append(row)
        return tuple(matches)

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
            self._db.execute(
                "UPDATE reservations SET status=?,decision='',reason=?,"
                "qualification_authority_digest='',qualification_evidence_digest='' "
                "WHERE reservation_id=?",
                (status, reason, reservation_id),
            )
        return self.get(reservation_id)


__all__ = [
    "FinalizedArrival", "FinalizedIntakeStore", "IntakeError", "IntakePolicy",
    "IntakeReservation",
]
