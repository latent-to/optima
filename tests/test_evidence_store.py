from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from optima.eval.evidence_store import (
    HARD_MAX_EVIDENCE_BYTES,
    EvidenceArtifactRef,
    EvidenceStoreError,
    prepare_evidence_root,
    publish_canonical_json_evidence,
    publish_evidence,
    reopen_evidence,
)
from optima.stack_identity import canonical_json_bytes


DOMAIN = "qualification.raw"
SCHEMA = "optima.qualification.raw.v1"
MEDIA = "application/vnd.optima.qualification+json"


def _target(root: Path, reference: EvidenceArtifactRef) -> Path:
    return root / reference.domain / reference.sha256[:2] / reference.sha256


def _publish(root: Path, payload: bytes = b"sealed evidence") -> EvidenceArtifactRef:
    return publish_evidence(
        root,
        payload,
        domain=DOMAIN,
        media_type=MEDIA,
        schema=SCHEMA,
    )


def test_reference_is_strict_and_round_trips() -> None:
    payload = b"abc"
    reference = EvidenceArtifactRef(
        DOMAIN, hashlib.sha256(payload).hexdigest(), len(payload), MEDIA, SCHEMA
    )
    assert EvidenceArtifactRef.from_dict(reference.to_dict()) == reference
    with pytest.raises(EvidenceStoreError, match="not closed"):
        EvidenceArtifactRef.from_dict({**reference.to_dict(), "headline": 9})
    with pytest.raises(EvidenceStoreError, match="not closed"):
        EvidenceArtifactRef.from_dict([reference.to_dict()])


@pytest.mark.parametrize(
    "changes",
    (
        {"domain": "../escape"},
        {"sha256": "A" * 64},
        {"size": True},
        {"size": HARD_MAX_EVIDENCE_BYTES + 1},
        {"media_type": "application/json; charset=utf-8"},
        {"schema": "../schema"},
    ),
)
def test_reference_rejects_noncanonical_fields(changes) -> None:
    values = dict(
        domain=DOMAIN,
        sha256="a" * 64,
        size=1,
        media_type=MEDIA,
        schema=SCHEMA,
    )
    values.update(changes)
    with pytest.raises(EvidenceStoreError):
        EvidenceArtifactRef(**values)


def test_prepare_root_is_private_owned_and_idempotent(tmp_path: Path) -> None:
    root = prepare_evidence_root(tmp_path / "evidence")
    assert root == prepare_evidence_root(root)
    info = root.lstat()
    assert stat.S_ISDIR(info.st_mode)
    assert stat.S_IMODE(info.st_mode) == 0o700
    assert info.st_uid == os.geteuid()


def test_prepare_rejects_relative_symlink_and_unsafe_roots(tmp_path: Path) -> None:
    with pytest.raises(EvidenceStoreError, match="canonical and absolute"):
        prepare_evidence_root(Path("relative/evidence"))

    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(EvidenceStoreError, match="unsafe|symlink"):
        prepare_evidence_root(link)

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o755)
    with pytest.raises(EvidenceStoreError, match="owner or mode"):
        prepare_evidence_root(unsafe)


def test_publish_reopen_and_exact_duplicate_are_content_addressed(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    payload = b"opaque\x00bytes"
    first = _publish(root, payload)
    target = _target(root, first)
    before = target.lstat()
    second = _publish(root, payload)
    after = target.lstat()

    assert first == second
    assert first.sha256 == hashlib.sha256(payload).hexdigest()
    assert first.size == len(payload)
    assert reopen_evidence(root, first) == payload
    assert before.st_ino == after.st_ino
    assert after.st_nlink == 1
    assert after.st_uid == os.geteuid()
    assert stat.S_IMODE(after.st_mode) == 0o400
    assert not list(target.parent.glob(".*.tmp.*"))


def test_publish_enforces_type_and_size_caps(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    with pytest.raises(EvidenceStoreError, match="exact bytes"):
        publish_evidence(
            root, bytearray(b"x"), domain=DOMAIN, media_type=MEDIA, schema=SCHEMA
        )
    with pytest.raises(EvidenceStoreError, match="size limit"):
        publish_evidence(
            root, b"xx", domain=DOMAIN, media_type=MEDIA, schema=SCHEMA, max_bytes=1
        )
    reference = _publish(root, b"xx")
    with pytest.raises(EvidenceStoreError, match="reference exceeds"):
        reopen_evidence(root, reference, max_bytes=1)


def test_canonical_json_is_convenience_not_semantic_authority(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    reference = publish_canonical_json_evidence(
        root,
        {"z": [2, 1], "a": "value"},
        domain=DOMAIN,
        schema=SCHEMA,
    )
    assert reopen_evidence(root, reference) == canonical_json_bytes(
        {"a": "value", "z": [2, 1]}
    )

    # A JSON media label does not make the byte store parse or approve JSON.
    opaque = publish_evidence(
        root,
        b"not json",
        domain=DOMAIN,
        media_type="application/json",
        schema=SCHEMA,
    )
    assert reopen_evidence(root, opaque) == b"not json"


def test_reopen_rejects_symlink_and_nonregular_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    reference = _publish(root)
    target = _target(root, reference)
    target.chmod(0o600)
    target.unlink()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_bytes(b"sealed evidence")
    target.symlink_to(elsewhere)
    with pytest.raises(EvidenceStoreError, match="unsafe shape"):
        reopen_evidence(root, reference)

    target.unlink()
    target.mkdir(mode=0o400)
    with pytest.raises(EvidenceStoreError, match="unsafe shape"):
        reopen_evidence(root, reference)


def test_reopen_rejects_hardlinks_and_unsafe_file_mode(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    reference = _publish(root)
    target = _target(root, reference)
    peer = tmp_path / "peer"
    os.link(target, peer)
    with pytest.raises(EvidenceStoreError, match="unsafe shape"):
        reopen_evidence(root, reference)
    peer.unlink()

    target.chmod(0o600)
    with pytest.raises(EvidenceStoreError, match="unsafe shape"):
        reopen_evidence(root, reference)


def test_reopen_rejects_truncation_growth_and_digest_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    payload = b"sealed evidence"
    reference = _publish(root, payload)
    target = _target(root, reference)

    target.chmod(0o600)
    target.write_bytes(payload[:-1])
    target.chmod(0o400)
    with pytest.raises(EvidenceStoreError, match="size"):
        reopen_evidence(root, reference)

    target.chmod(0o600)
    target.write_bytes(b"X" * len(payload))
    target.chmod(0o400)
    with pytest.raises(EvidenceStoreError, match="digest mismatch"):
        reopen_evidence(root, reference)

    wrong_size = replace(reference, size=reference.size - 1)
    with pytest.raises(EvidenceStoreError, match="size"):
        reopen_evidence(root, wrong_size)


def test_reopen_rejects_unsafe_directory_and_root_owner(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "evidence"
    reference = _publish(root)
    domain = root / DOMAIN
    domain.chmod(0o755)
    with pytest.raises(EvidenceStoreError, match="owner or mode"):
        reopen_evidence(root, reference)
    domain.chmod(0o700)

    monkeypatch.setattr(os, "geteuid", lambda: root.lstat().st_uid + 1)
    with pytest.raises(EvidenceStoreError, match="owner or mode"):
        reopen_evidence(root, reference)


def test_reopen_rejects_path_escape_through_store_symlink(tmp_path: Path) -> None:
    real = prepare_evidence_root(tmp_path / "real")
    reference = _publish(real)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(EvidenceStoreError, match="unsafe|symlink"):
        reopen_evidence(alias, reference)


def test_atomic_publish_failure_leaves_no_partial_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    import optima.eval.evidence_store as store

    root = tmp_path / "evidence"

    def fail(*_args, **_kwargs):
        raise OSError("injected rename failure")

    monkeypatch.setattr(store.os, "link", fail)
    with pytest.raises(EvidenceStoreError, match="cannot publish"):
        _publish(root)
    assert not any(path.is_file() for path in root.rglob("*"))


def test_reopen_requires_exact_reference_type(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    reference = _publish(root)
    with pytest.raises(EvidenceStoreError, match="exact and typed"):
        reopen_evidence(root, reference.to_dict())  # type: ignore[arg-type]
