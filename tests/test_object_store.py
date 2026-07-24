"""Tests for provider-swappable object storage and remote weight-offer publish."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from optima.chain.weight_share import (
    load_current_weight_offer_from_store,
    object_store_offer_loader,
    publish_current_weight_offer,
    read_current_weight_offer,
)
from optima.chain.weights import WeightProjection
from optima.object_store import (
    LocalDirectoryObjectStore,
    MemoryObjectStore,
    ObjectStoreConfig,
    ObjectStoreError,
    open_configured_object_store,
    open_object_store,
    prefixed_store,
)
from optima.stack_identity import canonical_digest, sha256_hex


def _d(label: str) -> str:
    return sha256_hex(label.encode())


def _projection(*, hotkey: str = "authority") -> WeightProjection:
    scope = _d("scope")
    metagraph_digest = canonical_digest(
        "optima.economics.metagraph-membership",
        {
            "block": 10,
            "block_hash": "0x" + f"{10:064x}",
            "chain_scope_digest": scope,
            "members": [
                {"hotkey": "authority", "uid": 0},
                {"hotkey": "miner", "uid": 1},
            ],
        },
    )
    return WeightProjection(
        scope,
        307,
        hotkey,
        _d("policy"),
        _d("settlement"),
        _d("evaluation"),
        metagraph_digest,
        (_d("arena"),),
        1,
        10,
        1,
        (_d("evidence"),),
        (("miner", 1_000_000),),
    )


def test_provider_presets_are_swappable_by_name() -> None:
    for provider in ("hippius", "s3", "minio"):
        cfg = ObjectStoreConfig(provider=provider, bucket="weights")
        assert cfg.provider == provider
    local = ObjectStoreConfig(provider="local", root_dir="/tmp/optima-store")
    assert local.provider == "local"
    with pytest.raises(ObjectStoreError, match="provider"):
        ObjectStoreConfig(provider="gpl-forbidden", bucket="x")


def test_memory_and_local_backends_roundtrip(tmp_path: Path) -> None:
    mem = MemoryObjectStore()
    mem.put_bytes("a.json", b'{"ok":true}\n', content_type="application/json")
    assert mem.get_bytes("a.json") == b'{"ok":true}\n'

    local = LocalDirectoryObjectStore(tmp_path / "root")
    local.put_bytes("dir/b.json", b"hello")
    assert local.get_bytes("dir/b.json") == b"hello"
    with pytest.raises(ObjectStoreError, match="missing"):
        local.get_bytes("missing.json")


def test_prefixed_store_and_open_configured(tmp_path: Path) -> None:
    cfg = ObjectStoreConfig(
        provider="local",
        root_dir=str(tmp_path / "root"),
        key_prefix="netuid/307",
    )
    store = open_configured_object_store(cfg)
    store.put_bytes("current_weights.json", b"payload")
    raw = (tmp_path / "root" / "netuid" / "307" / "current_weights.json").read_bytes()
    assert raw == b"payload"
    assert store.get_bytes("current_weights.json") == b"payload"


def test_publish_writes_local_and_remote_sync(tmp_path: Path) -> None:
    projection = _projection()
    local_path = tmp_path / "offer.json"
    remote = MemoryObjectStore()
    publish_current_weight_offer(
        projection,
        local_path=local_path,
        remote_store=remote,
        remote_key="current_weights.json",
        async_remote=False,
    )
    assert read_current_weight_offer(local_path).digest == projection.digest
    assert load_current_weight_offer_from_store(remote).digest == projection.digest


def test_publish_async_remote_does_not_block_local(tmp_path: Path) -> None:
    projection = _projection()
    local_path = tmp_path / "offer.json"
    started = threading.Event()
    released = threading.Event()

    class _SlowStore:
        def put_bytes(self, key, data, *, content_type="application/octet-stream"):
            started.set()
            assert released.wait(timeout=2)
            MemoryObjectStore().put_bytes(key, data, content_type=content_type)

        def get_bytes(self, key):
            raise AssertionError("unused")

    publish_current_weight_offer(
        projection,
        local_path=local_path,
        remote_store=_SlowStore(),
        async_remote=True,
    )
    # Local durability completes without waiting on the remote upload.
    assert read_current_weight_offer(local_path).digest == projection.digest
    assert started.wait(timeout=2)
    released.set()
    time.sleep(0.05)


def test_object_store_offer_loader(tmp_path: Path) -> None:
    projection = _projection()
    store = open_object_store(
        ObjectStoreConfig(provider="local", root_dir=str(tmp_path))
    )
    publish_current_weight_offer(
        projection,
        local_path=tmp_path / "local.json",
        remote_store=store,
        async_remote=False,
    )
    loaded = object_store_offer_loader(store)()
    assert loaded.digest == projection.digest


def test_hippius_to_s3_swap_is_config_only() -> None:
    hippius = ObjectStoreConfig(provider="hippius", bucket="optima-weights")
    aws = ObjectStoreConfig(
        provider="s3",
        bucket="optima-weights",
        region_name="us-west-2",
        endpoint_url=None,
    )
    minio = ObjectStoreConfig(
        provider="minio",
        bucket="optima-weights",
        endpoint_url="http://minio.internal:9000",
    )
    assert hippius.provider != aws.provider != minio.provider
    # Same logical key resolution regardless of provider.
    assert hippius.resolve_key("current_weights.json") == "current_weights.json"
    prefixed = ObjectStoreConfig(
        provider="hippius", bucket="optima-weights", key_prefix="prod/sn307"
    )
    assert prefixed.resolve_key("current_weights.json") == "prod/sn307/current_weights.json"


def test_open_s3_requires_boto3_message() -> None:
    cfg = ObjectStoreConfig(
        provider="hippius",
        bucket="optima-weights",
        access_key_id="hip_test",
        secret_access_key="secret",
    )
    try:
        import boto3  # noqa: F401
    except ImportError:
        with pytest.raises(ObjectStoreError, match="boto3"):
            open_object_store(cfg)
    else:
        # boto3 present in this environment: construction should at least reach client create.
        store = open_object_store(cfg)
        assert store.bucket == "optima-weights"
