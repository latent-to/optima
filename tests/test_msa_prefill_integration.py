"""Import-order and identity tests for the SGLang MSA prefill seam."""

from __future__ import annotations

import importlib.machinery
import sys
from types import ModuleType

import pytest

from optima.integrations import sglang_msa_prefill as seam


def _module(name: str) -> ModuleType:
    return ModuleType(name)


def _stock(*_args, **_kwargs):
    return "stock"


@pytest.fixture()
def fake_dispatcher(monkeypatch):
    made: list[tuple[object, object, object]] = []

    def factory(original, source, *, registry):
        def dispatcher(*args, **kwargs):
            return original(*args, **kwargs)

        made.append((original, source, registry))
        return dispatcher

    monkeypatch.setattr(seam, "make_msa_prefill_dispatcher", factory)
    return made


def _load_source(monkeypatch) -> ModuleType:
    source = _module(seam._SOURCE_MODULE)
    source.flash_prefill_with_topk_index = _stock
    monkeypatch.setitem(sys.modules, seam._SOURCE_MODULE, source)
    monkeypatch.delitem(sys.modules, seam._CONSUMER_MODULE, raising=False)
    return source


def test_already_imported_consumer_is_rebound_and_uninstalled(
    monkeypatch, fake_dispatcher
):
    source = _load_source(monkeypatch)
    consumer = _module(seam._CONSUMER_MODULE)
    consumer.flash_prefill_with_topk_index = _stock
    monkeypatch.setitem(sys.modules, seam._CONSUMER_MODULE, consumer)

    registry = object()
    seam.install(registry)
    dispatcher = source.flash_prefill_with_topk_index

    assert dispatcher is consumer.flash_prefill_with_topk_index
    assert dispatcher is not _stock
    assert fake_dispatcher == [(_stock, source, registry)]
    assert seam.is_installed()

    seam.install(registry)
    assert fake_dispatcher == [(_stock, source, registry)]
    assert source.flash_prefill_with_topk_index is dispatcher

    seam.uninstall()
    assert source.flash_prefill_with_topk_index is _stock
    assert consumer.flash_prefill_with_topk_index is _stock
    assert not seam.is_installed()


def test_source_first_makes_future_from_import_read_dispatcher(
    monkeypatch, fake_dispatcher
):
    source = _load_source(monkeypatch)
    seam.install()
    dispatcher = source.flash_prefill_with_topk_index

    # This assignment is the semantic result of a later
    # ``from flash_with_topk_idx import flash_prefill_with_topk_index``.
    consumer = _module(seam._CONSUMER_MODULE)
    consumer.flash_prefill_with_topk_index = source.flash_prefill_with_topk_index
    monkeypatch.setitem(sys.modules, seam._CONSUMER_MODULE, consumer)

    assert consumer.flash_prefill_with_topk_index is dispatcher
    assert seam.is_installed()
    seam.install()
    assert consumer.flash_prefill_with_topk_index is dispatcher

    seam.uninstall()
    assert source.flash_prefill_with_topk_index is _stock
    assert consumer.flash_prefill_with_topk_index is _stock


def test_consumer_mid_import_is_not_mistaken_for_upstream_drift(
    monkeypatch, fake_dispatcher
):
    source = _load_source(monkeypatch)
    consumer = _module(seam._CONSUMER_MODULE)
    spec = importlib.machinery.ModuleSpec(seam._CONSUMER_MODULE, loader=None)
    spec._initializing = True
    consumer.__spec__ = spec
    monkeypatch.setitem(sys.modules, seam._CONSUMER_MODULE, consumer)

    seam.install()
    dispatcher = source.flash_prefill_with_topk_index
    assert seam.is_installed()

    # The paused ``from`` statement resumes after the source post-import hook.
    consumer.flash_prefill_with_topk_index = source.flash_prefill_with_topk_index
    spec._initializing = False
    assert consumer.flash_prefill_with_topk_index is dispatcher
    assert seam.is_installed()


def test_foreign_loaded_consumer_fails_atomically(monkeypatch, fake_dispatcher):
    source = _load_source(monkeypatch)
    consumer = _module(seam._CONSUMER_MODULE)
    foreign = lambda: None
    consumer.flash_prefill_with_topk_index = foreign
    monkeypatch.setitem(sys.modules, seam._CONSUMER_MODULE, consumer)

    with pytest.raises(RuntimeError, match="refusing to clobber"):
        seam.install()

    assert source.flash_prefill_with_topk_index is _stock
    assert consumer.flash_prefill_with_topk_index is foreign
    assert not getattr(source, seam._PATCH_FLAG, False)
    assert not hasattr(source, seam._ORIG_ATTR)
    assert not hasattr(source, seam._DISPATCH_ATTR)


def test_binding_drift_is_not_reported_installed_or_clobbered_on_uninstall(
    monkeypatch, fake_dispatcher
):
    source = _load_source(monkeypatch)
    consumer = _module(seam._CONSUMER_MODULE)
    consumer.flash_prefill_with_topk_index = _stock
    monkeypatch.setitem(sys.modules, seam._CONSUMER_MODULE, consumer)
    seam.install()
    dispatcher = source.flash_prefill_with_topk_index

    foreign = lambda: None
    consumer.flash_prefill_with_topk_index = foreign
    assert not seam.is_installed()
    with pytest.raises(RuntimeError, match="refusing to clobber"):
        seam.install()
    with pytest.raises(RuntimeError, match="refusing to clobber"):
        seam.uninstall()

    assert source.flash_prefill_with_topk_index is dispatcher
    assert consumer.flash_prefill_with_topk_index is foreign
    assert getattr(source, seam._PATCH_FLAG) is True


def test_completed_consumer_without_call_site_fails_closed(
    monkeypatch, fake_dispatcher
):
    source = _load_source(monkeypatch)
    consumer = _module(seam._CONSUMER_MODULE)
    monkeypatch.setitem(sys.modules, seam._CONSUMER_MODULE, consumer)

    with pytest.raises(RuntimeError, match="unreachable seam"):
        seam.install()
    assert source.flash_prefill_with_topk_index is _stock
    assert not seam.is_installed()
