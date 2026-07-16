from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from optima.model_provision import (
    ModelProvisionError,
    provision_model,
    reopen_model_provision,
)


def _tree(root: Path) -> None:
    (root / "weights").mkdir(parents=True)
    (root / "config.json").write_bytes(b'{"model":"fixture"}\n')
    (root / "weights" / "b.bin").write_bytes(b"bbb")
    (root / "weights" / "a.bin").write_bytes(b"aaa")
    (root / ".hidden").write_bytes(b"included")


def test_provision_is_deterministic_across_workers_and_roots(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    receipts = tmp_path / "receipts"
    first_root.mkdir()
    second_root.mkdir()
    receipts.mkdir()
    _tree(first_root)
    _tree(second_root)

    first = provision_model(first_root, receipts, workers=1)
    second = provision_model(second_root, receipts, workers=4)

    assert first.receipt == second.receipt
    assert first.receipt_path == second.receipt_path
    assert first.receipt_path.read_bytes() == first.receipt.canonical_bytes
    assert [row.path for row in first.receipt.files] == [
        ".hidden",
        "config.json",
        "weights/a.bin",
        "weights/b.bin",
    ]
    reopened = reopen_model_provision(
        second_root,
        second.receipt_path,
        expected_content_digest=second.receipt.content_digest,
        expected_receipt_digest=second.receipt.receipt_digest,
        workers=2,
    )
    assert reopened == second


def test_wrong_expected_digest_publishes_nothing(tmp_path: Path) -> None:
    model = tmp_path / "model"
    receipts = tmp_path / "receipts"
    model.mkdir()
    receipts.mkdir()
    _tree(model)

    with pytest.raises(ModelProvisionError, match="expected_content_digest"):
        provision_model(model, receipts, expected_content_digest="0" * 64)

    assert list(receipts.iterdir()) == []


def test_provision_rejects_transient_cache_tree(tmp_path: Path) -> None:
    model = tmp_path / "model"
    receipts = tmp_path / "receipts"
    model.mkdir()
    receipts.mkdir()
    _tree(model)
    metadata = model / ".cache" / "huggingface" / "download" / "config.json.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("etag\ncommit\ntimestamp\n")

    with pytest.raises(ModelProvisionError, match="transient cache path: \\.cache"):
        provision_model(model, receipts, workers=1)

    assert list(receipts.iterdir()) == []


def test_reopen_rejects_transient_cache_added_after_provision(tmp_path: Path) -> None:
    model = tmp_path / "model"
    receipts = tmp_path / "receipts"
    model.mkdir()
    receipts.mkdir()
    _tree(model)
    published = provision_model(model, receipts, workers=1)
    cache = model / ".cache"
    cache.mkdir()
    (cache / "metadata").write_text("mutable\n")

    with pytest.raises(ModelProvisionError, match="transient cache path: \\.cache"):
        reopen_model_provision(model, published.receipt_path, workers=1)


@pytest.mark.parametrize("kind", ["root_symlink", "nested_symlink", "hardlink", "fifo"])
def test_provision_rejects_nonconcrete_or_aliased_trees(
    tmp_path: Path, kind: str
) -> None:
    model = tmp_path / "model"
    receipts = tmp_path / "receipts"
    model.mkdir()
    receipts.mkdir()
    (model / "file").write_bytes(b"data")
    root: Path = model
    if kind == "root_symlink":
        root = tmp_path / "model-link"
        root.symlink_to(model, target_is_directory=True)
    elif kind == "nested_symlink":
        (model / "link").symlink_to(model / "file")
    elif kind == "hardlink":
        os.link(model / "file", model / "alias")
    else:
        os.mkfifo(model / "fifo")

    with pytest.raises(ModelProvisionError):
        provision_model(root, receipts, workers=1)


def test_receipt_publication_is_idempotent_but_never_clobbers(tmp_path: Path) -> None:
    model = tmp_path / "model"
    receipts = tmp_path / "receipts"
    model.mkdir()
    receipts.mkdir()
    _tree(model)
    published = provision_model(model, receipts, workers=2)
    original = published.receipt_path.read_bytes()
    assert provision_model(model, receipts, workers=1) == published

    published.receipt_path.chmod(0o644)
    published.receipt_path.write_bytes(b"{}\n")
    with pytest.raises(ModelProvisionError):
        provision_model(model, receipts, workers=1)
    assert published.receipt_path.read_bytes() == b"{}\n"
    assert published.receipt_path.read_bytes() != original


@pytest.mark.parametrize("mutation", ["bytes", "extra", "missing"])
def test_reopen_rehashes_the_complete_tree(tmp_path: Path, mutation: str) -> None:
    model = tmp_path / "model"
    receipts = tmp_path / "receipts"
    model.mkdir()
    receipts.mkdir()
    _tree(model)
    published = provision_model(model, receipts, workers=2)
    if mutation == "bytes":
        (model / "config.json").write_bytes(b'{"model":"changed"}\n')
    elif mutation == "extra":
        (model / "unreceipted.bin").write_bytes(b"extra")
    else:
        (model / "weights" / "a.bin").unlink()

    with pytest.raises(ModelProvisionError, match="does not match"):
        reopen_model_provision(model, published.receipt_path, workers=2)


@pytest.mark.parametrize("attack", ["symlink", "hardlink", "noncanonical"])
def test_reopen_rejects_unsafe_or_noncanonical_publication(
    tmp_path: Path, attack: str
) -> None:
    model = tmp_path / "model"
    receipts = tmp_path / "receipts"
    model.mkdir()
    receipts.mkdir()
    _tree(model)
    published = provision_model(model, receipts, workers=1)
    original = published.receipt_path.read_bytes()
    published.receipt_path.chmod(0o644)
    published.receipt_path.unlink()
    if attack == "symlink":
        outside = tmp_path / "outside.json"
        outside.write_bytes(original)
        published.receipt_path.symlink_to(outside)
    elif attack == "hardlink":
        outside = tmp_path / "outside.json"
        outside.write_bytes(original)
        os.link(outside, published.receipt_path)
    else:
        value = json.loads(original)
        published.receipt_path.write_text(json.dumps(value, indent=2) + "\n")

    with pytest.raises(ModelProvisionError):
        reopen_model_provision(model, published.receipt_path, workers=1)


def test_publication_directory_must_not_be_inside_model_tree(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _tree(model)
    receipts = model / "receipts"
    receipts.mkdir()

    with pytest.raises(ModelProvisionError, match="outside the model tree"):
        provision_model(model, receipts, workers=1)
