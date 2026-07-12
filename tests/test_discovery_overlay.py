from __future__ import annotations

import hashlib
import multiprocessing.process
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import optima.discovery_overlay as overlay_module

from optima.discovery_overlay import (
    ACTIVE_IDENTITY,
    ARMED,
    DISCOVERY_ENVIRONMENT_KEYS,
    DRIVER_PID,
    EXPECTED_IDENTITY,
    PROCESS_ROLE,
    ROLE_PARENT_PID,
    DiscoveryActivationReceipt,
    DiscoveryOverlayActivationError,
    activation_policy_digest,
    activate_scheduler_overlay,
    arm_driver_activation,
    clear_driver_activation,
    install_process_role_hook,
    launch_environment,
    require_driver_activation,
    require_stock_driver_origin,
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def test_activation_policy_identity_is_fixed_and_content_addressed() -> None:
    digest = activation_policy_digest()
    assert digest == activation_policy_digest()
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def _scheduler_target() -> None:
    raise AssertionError("the fake scheduler target must not execute")


_scheduler_target.__module__ = "sglang.srt.managers.scheduler"
_scheduler_target.__qualname__ = "run_scheduler_process"


class _Distribution:
    def __init__(self, root: Path, version: str = "0.0.0.dev1") -> None:
        self.root = root
        self.version = version

    def locate_file(self, relative: str) -> Path:
        return self.root / relative


class _FakeProcess:
    def __init__(
        self,
        *,
        server_args: object,
        pid: int,
        gpu_id: int,
        tp_rank: int,
        pp_rank: int = 0,
        dp_rank: int | None = None,
        alive: bool = True,
    ) -> None:
        self._target = _scheduler_target
        self._args = (
            server_args,
            object(),
            gpu_id,
            tp_rank,
            tp_rank,
            0,
            tp_rank,
            pp_rank,
            dp_rank,
            object(),
        )
        self._kwargs: dict[str, object] = {}
        self._start_method = "spawn"
        self.pid = pid
        self.exitcode = None if alive else 1
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


@pytest.fixture(autouse=True)
def _clear_activation_state():
    clear_driver_activation()
    yield
    clear_driver_activation()


def _server_args(tp_size: int, **overrides: int) -> object:
    values = {
        "dp_size": 1,
        "nnodes": 1,
        "node_rank": 0,
        "pp_size": 1,
        "tp_size": tp_size,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _stock_module(tmp_path: Path, *, version: str = "0.0.0.dev1"):
    root = tmp_path / "installed"
    package = root / "sglang"
    package.mkdir(parents=True)
    module_file = package / "__init__.py"
    module_file.write_text("__version__ = 'stock'\n")
    module = SimpleNamespace(
        __file__=str(module_file),
        __name__="sglang",
        __spec__=SimpleNamespace(name="sglang", origin=str(module_file)),
    )
    return module, _Distribution(root, version), root


def _overlay_root(tmp_path: Path) -> Path:
    root = tmp_path / "overlay"
    (root / "site" / "sglang").mkdir(parents=True)
    return root


def _arm_environment(monkeypatch, root: Path, identity: str, members: int) -> None:
    environment = launch_environment(
        overlay_root=root,
        expected_identity_digest=identity,
        driver_pid=os.getpid(),
    )
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    arm_driver_activation(
        expected_identity_digest=identity,
        expected_members=members,
    )


def _install_fake_start(monkeypatch, seen: list[tuple[str, str]]) -> object:
    def original(process, *args, **kwargs):
        seen.append((os.environ.get(PROCESS_ROLE, ""), os.environ.get(ROLE_PARENT_PID, "")))
        return "started"

    monkeypatch.setattr(multiprocessing.process.BaseProcess, "start", original)
    install_process_role_hook()
    return multiprocessing.process.BaseProcess.start


def test_scheduler_spawn_substitutes_trampoline_then_restores_target(
    tmp_path, monkeypatch
):
    identity = _h("overlay")
    root = _overlay_root(tmp_path)
    _arm_environment(monkeypatch, root, identity, 1)
    observed: list[tuple[str, str]] = []

    def original(process, *args, **kwargs):
        target = process._target
        observed.append((target.__module__, target.__qualname__))
        return "started"

    monkeypatch.setattr(multiprocessing.process.BaseProcess, "start", original)
    install_process_role_hook()
    process = _FakeProcess(
        server_args=_server_args(1), pid=351, gpu_id=0, tp_rank=0
    )
    assert multiprocessing.process.BaseProcess.start(process) == "started"
    assert observed == [
        ("optima.discovery_overlay", "_scheduler_overlay_entry")
    ]
    assert process._target is _scheduler_target


def test_scheduler_trampoline_requires_active_overlay_before_calling_target(
    monkeypatch,
):
    identity = _h("overlay")
    monkeypatch.setenv(EXPECTED_IDENTITY, identity)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def target(*args, **kwargs):
        calls.append((args, kwargs))
        return "ran"

    target.__module__ = "sglang.srt.managers.scheduler"
    target.__qualname__ = "run_scheduler_process"
    module = SimpleNamespace(run_scheduler_process=target)
    monkeypatch.setattr(
        overlay_module.importlib, "import_module", lambda name: module
    )
    monkeypatch.setattr(overlay_module, "install", lambda: None)

    with pytest.raises(DiscoveryOverlayActivationError, match="did not activate"):
        overlay_module._scheduler_overlay_entry("first")
    assert calls == []

    monkeypatch.setenv(ACTIVE_IDENTITY, identity)
    assert overlay_module._scheduler_overlay_entry("first", rank=2) == "ran"
    assert calls == [(('first',), {"rank": 2})]


def test_launch_environment_clears_every_inactive_marker(tmp_path):
    identity = _h("overlay")
    environment = launch_environment(
        overlay_root=_overlay_root(tmp_path),
        expected_identity_digest=identity,
        driver_pid=17,
    )

    assert set(environment) == set(DISCOVERY_ENVIRONMENT_KEYS)
    assert environment[ARMED] == "1"
    assert environment[DRIVER_PID] == "17"
    assert environment[EXPECTED_IDENTITY] == identity
    assert environment[PROCESS_ROLE] == ""
    assert environment[ROLE_PARENT_PID] == ""
    assert environment[ACTIVE_IDENTITY] == ""


def test_injected_overlay_reader_is_not_passed_a_none_read_only_callback(tmp_path):
    identity = _h("overlay")
    root = _overlay_root(tmp_path)
    environment = launch_environment(
        overlay_root=root,
        expected_identity_digest=identity,
        driver_pid=11,
    )
    environment[PROCESS_ROLE] = "scheduler"
    environment[ROLE_PARENT_PID] = "11"
    calls: list[tuple[Path, str, bool]] = []

    def reader(path, *, expected_identity_digest, require_read_only):
        calls.append((path, expected_identity_digest, require_read_only))
        return SimpleNamespace(
            identity=SimpleNamespace(digest=identity),
            root=root,
        )

    paths: list[str] = []
    activated = activate_scheduler_overlay(
        environment=environment,
        pid=12,
        parent_pid=11,
        modules={},
        sys_path=paths,
        reader=reader,
    )

    assert activated == (root / "site", identity)
    assert calls == [(root, identity, True)]
    assert paths == [str(root / "site")]


def test_stock_driver_ledger_produces_one_tp_complete_data_receipt(
    tmp_path, monkeypatch
):
    identity = _h("overlay")
    root = _overlay_root(tmp_path)
    _arm_environment(monkeypatch, root, identity, 2)
    seen: list[tuple[str, str]] = []
    start = _install_fake_start(monkeypatch, seen)
    first_pid = os.getpid() + 100
    second_pid = first_pid + 1

    assert start(_FakeProcess(
        server_args=_server_args(2), pid=first_pid, gpu_id=4, tp_rank=0
    )) == "started"
    assert start(_FakeProcess(
        server_args=_server_args(2), pid=second_pid, gpu_id=7, tp_rank=1
    )) == "started"
    module, distribution, stock_root = _stock_module(tmp_path)
    receipt = require_driver_activation(
        module,
        root,
        expected_identity_digest=identity,
        expected_members=2,
        expected_sglang_version=distribution.version,
        distribution=distribution,
        search_path=[str(stock_root)],
    )

    assert type(receipt) is DiscoveryActivationReceipt
    assert receipt.overlay_identity_digest == identity
    assert receipt.driver_pid == os.getpid()
    assert receipt.tp_size == 2
    assert tuple(row.tp_rank for row in receipt.members) == (0, 1)
    assert tuple(row.pid for row in receipt.members) == (first_pid, second_pid)
    assert tuple(row.gpu_id for row in receipt.members) == (4, 7)
    assert receipt.driver_origin.to_dict() == {
        "distribution": "sglang",
        "module": "sglang/__init__.py",
        "version": distribution.version,
    }
    assert receipt.to_dict()["members"] == [
        row.to_dict() for row in receipt.members
    ]
    assert DiscoveryActivationReceipt.from_dict(receipt.to_dict()) == receipt
    assert seen == [("scheduler", str(os.getpid()))] * 2
    assert os.environ[PROCESS_ROLE] == ""
    assert os.environ[ROLE_PARENT_PID] == ""


def test_scheduler_written_files_cannot_satisfy_driver_activation(
    tmp_path, monkeypatch
):
    identity = _h("overlay")
    root = _overlay_root(tmp_path)
    (root / "active.999.json").write_text(
        '{"identity_digest":"' + identity + '","pid":999}'
    )
    _arm_environment(monkeypatch, root, identity, 1)
    module, distribution, stock_root = _stock_module(tmp_path)

    with pytest.raises(DiscoveryOverlayActivationError, match="incomplete \\(0/1\\)"):
        require_driver_activation(
            module,
            root,
            expected_identity_digest=identity,
            expected_members=1,
            expected_sglang_version=distribution.version,
            distribution=distribution,
            search_path=[str(stock_root)],
        )


@pytest.mark.parametrize(
    ("override", "value"),
    (("dp_size", 2), ("pp_size", 2), ("nnodes", 2), ("node_rank", 1)),
)
def test_driver_ledger_rejects_topologies_it_cannot_prove(
    tmp_path, monkeypatch, override, value
):
    identity = _h("overlay")
    root = _overlay_root(tmp_path)
    _arm_environment(monkeypatch, root, identity, 1)
    seen: list[tuple[str, str]] = []
    start = _install_fake_start(monkeypatch, seen)
    process = _FakeProcess(
        server_args=_server_args(1, **{override: value}),
        pid=301,
        gpu_id=0,
        tp_rank=0,
    )

    with pytest.raises(DiscoveryOverlayActivationError, match="one-node DP1/PP1"):
        start(process)
    assert seen == []


def test_driver_ledger_rejects_incomplete_dead_and_duplicate_members(
    tmp_path, monkeypatch
):
    identity = _h("overlay")
    root = _overlay_root(tmp_path)
    _arm_environment(monkeypatch, root, identity, 2)
    start = _install_fake_start(monkeypatch, [])
    start(_FakeProcess(
        server_args=_server_args(2), pid=401, gpu_id=0, tp_rank=0, alive=False
    ))
    with pytest.raises(DiscoveryOverlayActivationError, match="duplicate"):
        start(_FakeProcess(
            server_args=_server_args(2), pid=402, gpu_id=1, tp_rank=0
        ))

    module, distribution, stock_root = _stock_module(tmp_path)
    with pytest.raises(DiscoveryOverlayActivationError, match="incomplete \\(1/2\\)"):
        require_driver_activation(
            module,
            root,
            expected_identity_digest=identity,
            expected_members=2,
            expected_sglang_version=distribution.version,
            distribution=distribution,
            search_path=[str(stock_root)],
        )


def test_driver_ledger_rejects_a_changed_pinned_scheduler_signature(
    tmp_path, monkeypatch
):
    identity = _h("overlay")
    root = _overlay_root(tmp_path)
    _arm_environment(monkeypatch, root, identity, 1)
    seen: list[tuple[str, str]] = []
    start = _install_fake_start(monkeypatch, seen)
    process = _FakeProcess(
        server_args=_server_args(1), pid=451, gpu_id=0, tp_rank=0
    )
    process._args = process._args[:-1]

    with pytest.raises(DiscoveryOverlayActivationError, match="signature changed"):
        start(process)
    assert seen == []


def test_driver_ledger_rejects_a_member_that_exits_before_ready(
    tmp_path, monkeypatch
):
    identity = _h("overlay")
    root = _overlay_root(tmp_path)
    _arm_environment(monkeypatch, root, identity, 1)
    start = _install_fake_start(monkeypatch, [])
    start(_FakeProcess(
        server_args=_server_args(1), pid=501, gpu_id=0, tp_rank=0, alive=False
    ))
    module, distribution, stock_root = _stock_module(tmp_path)

    with pytest.raises(DiscoveryOverlayActivationError, match="exited before"):
        require_driver_activation(
            module,
            root,
            expected_identity_digest=identity,
            expected_members=1,
            expected_sglang_version=distribution.version,
            distribution=distribution,
            search_path=[str(stock_root)],
        )


def test_stock_driver_origin_requires_distribution_path_version_and_clean_search_path(
    tmp_path,
):
    root = _overlay_root(tmp_path)
    module, distribution, stock_root = _stock_module(tmp_path)

    assert require_stock_driver_origin(
        module,
        root,
        expected_sglang_version=distribution.version,
        distribution=distribution,
        search_path=[str(stock_root)],
    ).module == "sglang/__init__.py"
    with pytest.raises(DiscoveryOverlayActivationError, match="exact pinned stock"):
        require_stock_driver_origin(
            module,
            root,
            expected_sglang_version="9.9.9",
            distribution=distribution,
            search_path=[str(stock_root)],
        )
    with pytest.raises(DiscoveryOverlayActivationError, match="exact pinned stock"):
        require_stock_driver_origin(
            module,
            root,
            expected_sglang_version=distribution.version,
            distribution=distribution,
            search_path=[str(root / "site")],
        )


def test_stock_driver_origin_resolves_pinned_editable_install(
    tmp_path, monkeypatch
):
    import importlib.util

    root = _overlay_root(tmp_path)
    module, _distribution, stock_root = _stock_module(tmp_path)
    metadata_root = tmp_path / "site-packages"
    metadata_root.mkdir()
    editable = _Distribution(metadata_root, _distribution.version)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(
            origin=module.__file__,
            submodule_search_locations=(str(stock_root / "sglang"),),
        ),
    )

    assert require_stock_driver_origin(
        module,
        root,
        expected_sglang_version=editable.version,
        distribution=editable,
        search_path=[str(stock_root)],
    ).module == "sglang/__init__.py"


def test_armed_scheduler_spawn_requires_an_explicit_driver_window(
    tmp_path, monkeypatch
):
    identity = _h("overlay")
    environment = launch_environment(
        overlay_root=_overlay_root(tmp_path),
        expected_identity_digest=identity,
        driver_pid=os.getpid(),
    )
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    start = _install_fake_start(monkeypatch, [])

    with pytest.raises(DiscoveryOverlayActivationError, match="outside the armed"):
        start(_FakeProcess(
            server_args=_server_args(1), pid=601, gpu_id=0, tp_rank=0
        ))
