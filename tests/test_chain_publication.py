from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest

from optima.bundle_hash import content_hash
from optima.chain.publication import (
    WorkerBundlePublicationError,
    publish_worker_bundle,
    reopen_worker_bundle,
)


def _private_bundle(root: Path) -> Path:
    root.mkdir(mode=0o700)
    kernels = root / "kernels"
    metadata = root / "metadata"
    kernels.mkdir(mode=0o700)
    metadata.mkdir(mode=0o700)
    files = {
        root / "manifest.toml": b'bundle_id = "worker-test"\n',
        kernels / "entry.py": b"def entry(x):\n    return x\n",
        metadata / "entry.json": b'{"dtype":"bfloat16"}\n',
    }
    for path, payload in files.items():
        path.write_bytes(payload)
        path.chmod(0o600)
    return root


def _all_modes(root: Path) -> tuple[set[int], set[int]]:
    directories: set[int] = set()
    files: set[int] = set()
    for path in root.rglob("*"):
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_dir():
            directories.add(mode)
        else:
            files.add(mode)
    return directories, files


def test_publish_and_reopen_worker_bundle_is_separate_immutable_and_typed(tmp_path):
    private = _private_bundle(tmp_path / "private")
    committed = content_hash(private)
    publications = tmp_path / "publications"

    published = publish_worker_bundle(private, publications, committed)

    assert published.root != private
    assert private not in published.root.parents
    assert published.content_hash == committed
    assert published.root.name == published.address_digest
    assert published.files
    directories, files = _all_modes(published.root)
    assert directories == {0o555}
    assert files == {0o444}
    # Read/execute bits required for a non-owner worker remain present after sealing.
    assert stat.S_IMODE(published.root.stat().st_mode) & 0o055 == 0o055
    assert all(
        stat.S_IMODE(
            published.root.joinpath(*row.path.split("/")).stat().st_mode
        )
        & 0o044
        == 0o044
        for row in published.files
    )

    reopened = reopen_worker_bundle(
        published.root,
        committed,
        expected_publication_digest=published.publication_digest,
        expected_receipt_digest=published.digest,
    )
    assert reopened == published
    assert reopened.to_dict() == published.to_dict()
    # OCI-side reopen may relax host ownership only; canonical paths, link shape,
    # 0444/0555 modes, complete inventory, and both digests remain mandatory.
    assert reopen_worker_bundle(
        published.root,
        committed,
        expected_publication_digest=published.publication_digest,
        expected_owner_uid=None,
    ).digest == published.digest


def test_publish_reuses_only_the_exact_same_committed_tree(tmp_path):
    private = _private_bundle(tmp_path / "private")
    committed = content_hash(private)
    publications = tmp_path / "publications"

    first = publish_worker_bundle(private, publications, committed)
    second = publish_worker_bundle(private, publications, committed)

    assert second.root == first.root
    assert second.digest == first.digest
    assert second.reused is True


def test_publish_rejects_wrong_committed_hash_before_publication(tmp_path):
    private = _private_bundle(tmp_path / "private")
    publications = tmp_path / "publications"

    with pytest.raises(WorkerBundlePublicationError, match="committed content hash"):
        publish_worker_bundle(private, publications, "a" * 64)
    assert not publications.exists()


@pytest.mark.parametrize(
    "relative",
    (
        ".git/config",
        "__pycache__/entry.cpython-313.pyc",
        "kernels/entry.pyc",
        "kernels/entry.pyo",
        "kernels/._entry.py",
    ),
)
def test_publish_rejects_bytes_excluded_from_bundle_identity(tmp_path, relative):
    private = _private_bundle(tmp_path / "private")
    committed = content_hash(private)
    extra = private.joinpath(*relative.split("/"))
    extra.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    extra.parent.chmod(0o700)
    extra.write_bytes(b"uncommitted executable bytes")
    extra.chmod(0o600)
    # The legacy bundle hash deliberately ignores this path, proving the extra
    # would share the same chain commitment without the publication guard.
    assert content_hash(private) == committed

    with pytest.raises(WorkerBundlePublicationError, match="excluded"):
        publish_worker_bundle(private, tmp_path / "publications", committed)


def test_publish_rejects_unhashed_empty_directories(tmp_path):
    private = _private_bundle(tmp_path / "private")
    committed = content_hash(private)
    (private / "uncommitted-empty").mkdir(mode=0o700)
    assert content_hash(private) == committed

    with pytest.raises(WorkerBundlePublicationError, match="empty"):
        publish_worker_bundle(private, tmp_path / "publications", committed)


def test_publish_rejects_non_private_source_and_overlapping_roots(tmp_path):
    private = _private_bundle(tmp_path / "private")
    committed = content_hash(private)
    private.chmod(0o755)
    with pytest.raises(WorkerBundlePublicationError, match="0700"):
        publish_worker_bundle(private, tmp_path / "publications", committed)

    private.chmod(0o700)
    with pytest.raises(WorkerBundlePublicationError, match="overlap"):
        publish_worker_bundle(private, private / "published", committed)


def test_publish_rejects_symlink_hardlink_and_group_readable_source(tmp_path):
    private = _private_bundle(tmp_path / "private")
    committed = content_hash(private)
    entry = private / "kernels" / "entry.py"

    entry.chmod(0o640)
    with pytest.raises(WorkerBundlePublicationError, match="ownership/mode/link"):
        publish_worker_bundle(private, tmp_path / "mode-publications", committed)

    entry.chmod(0o600)
    hardlink = private / "kernels" / "hardlink.py"
    os.link(entry, hardlink)
    with pytest.raises(WorkerBundlePublicationError, match="ownership/mode/link"):
        publish_worker_bundle(
            private, tmp_path / "hardlink-publications", content_hash(private)
        )
    hardlink.unlink()

    symlink = private / "kernels" / "symlink.py"
    symlink.symlink_to(entry)
    with pytest.raises(WorkerBundlePublicationError):
        publish_worker_bundle(
            private, tmp_path / "symlink-publications", content_hash(private)
        )


def test_reopen_fails_closed_on_wrong_hash_receipt_or_mutated_publication(tmp_path):
    private = _private_bundle(tmp_path / "private")
    committed = content_hash(private)
    published = publish_worker_bundle(private, tmp_path / "publications", committed)

    with pytest.raises(WorkerBundlePublicationError):
        reopen_worker_bundle(published.root, "b" * 64)
    with pytest.raises(WorkerBundlePublicationError, match="receipt"):
        reopen_worker_bundle(
            published.root,
            committed,
            expected_publication_digest=published.publication_digest,
            expected_receipt_digest="c" * 64,
        )

    entry = published.root / "kernels" / "entry.py"
    entry.chmod(0o644)
    entry.write_bytes(b"changed")
    entry.chmod(0o444)
    with pytest.raises(WorkerBundlePublicationError, match="reopen failed"):
        reopen_worker_bundle(
            published.root,
            committed,
            expected_publication_digest=published.publication_digest,
        )


def test_publication_address_binds_path_inventory_not_only_file_bytes(tmp_path):
    left = _private_bundle(tmp_path / "left")
    right = _private_bundle(tmp_path / "right")
    old = right / "kernels" / "entry.py"
    new = right / "kernels" / "renamed.py"
    old.rename(new)
    # Reproduce the identity calculation explicitly to ensure the assertion does
    # not accidentally depend on hash object reuse.
    assert hashlib.sha256(new.read_bytes()).digest() == hashlib.sha256(
        (left / "kernels" / "entry.py").read_bytes()
    ).digest()

    first = publish_worker_bundle(left, tmp_path / "publications", content_hash(left))
    second = publish_worker_bundle(right, tmp_path / "publications", content_hash(right))
    assert first.address_digest != second.address_digest
    assert first.root != second.root
