"""Scheduler-only activation for a sealed discovery SGLang overlay.

The isolated engine driver must continue importing the image-pinned stock
package.  Only exact SGLang scheduler children may import the discovery tree.
The worker supplies a read-only overlay and its expected content identity; this
module never accepts a bundle path, build command, cache key, or miner-selected
environment.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import multiprocessing
import multiprocessing.process
import os
import pickle
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Sequence


ARMED = "OPTIMA_DISCOVERY_OVERLAY_ARMED"
PROCESS_ROLE = "OPTIMA_DISCOVERY_PROCESS_ROLE"
ROLE_PARENT_PID = "OPTIMA_DISCOVERY_ROLE_PARENT_PID"
DRIVER_PID = "OPTIMA_DISCOVERY_DRIVER_PID"
OVERLAY_ROOT = "OPTIMA_DISCOVERY_OVERLAY_ROOT"
EXPECTED_IDENTITY = "OPTIMA_DISCOVERY_EXPECTED_IDENTITY"
ACTIVE_IDENTITY = "OPTIMA_DISCOVERY_OVERLAY_ACTIVE"
ACTIVATION_FAILED = "OPTIMA_DISCOVERY_OVERLAY_FAILED"

DISCOVERY_ENVIRONMENT_KEYS = (
    ARMED,
    PROCESS_ROLE,
    ROLE_PARENT_PID,
    DRIVER_PID,
    OVERLAY_ROOT,
    EXPECTED_IDENTITY,
    ACTIVE_IDENTITY,
    ACTIVATION_FAILED,
)

_SCHEDULER_ROLE = "scheduler"
_SCHEDULER_MODULE = "sglang.srt.managers.scheduler"
_SCHEDULER_QUALNAME = "run_scheduler_process"
_SERVER_ARGS_MODULE = "sglang.srt.server_args"
_TRAMPOLINE_MODULE = "optima.discovery_overlay"
_TRAMPOLINE_QUALNAME = "_scheduler_overlay_entry"
_SCHEDULER_ARGUMENT_SCHEMA = "optima.discovery-scheduler-arguments.v1"
_SCHEDULER_PICKLE_PROTOCOL = 5
_MAX_SCHEDULER_ARGUMENT_BYTES = 1 << 20
_DRIVER_MODULE = "sglang/__init__.py"
_ACTIVATION_SCHEMA = "optima.discovery-driver-activation.v1"
_ACTIVATION_POLICY_SCHEMA = "optima.discovery-activation-policy.v1"
_MAX_MEMBERS = 64
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_VERSION = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,127}\Z")
_ENV_SPAWN_LOCK = threading.RLock()


class DiscoveryOverlayActivationError(RuntimeError):
    """A sealed overlay could not be activated at the scheduler boundary."""


def activation_policy_digest() -> str:
    """Identify the fixed stock-driver predicates behind an activation receipt."""

    payload = {
        "coverage": {
            "live_at_ready": True,
            "tp_ranks": "complete-0-to-tp-minus-one",
            "unique": ["gpu_id", "pid", "tp_rank"],
        },
        "driver": {
            "distribution": "sglang",
            "module": _DRIVER_MODULE,
            "overlay_visible": False,
            "version_bound_by_requirement": True,
        },
        "scheduler": {
            "argument_transport": {
                "custom_sigquit_handler": None,
                "max_pickle_bytes": _MAX_SCHEDULER_ARGUMENT_BYTES,
                "parent_types": [
                    f"{_SERVER_ARGS_MODULE}.PortArgs",
                    f"{_SERVER_ARGS_MODULE}.ServerArgs",
                ],
                "pickle_protocol": _SCHEDULER_PICKLE_PROTOCOL,
                "schema": _SCHEDULER_ARGUMENT_SCHEMA,
            },
            "dp_rank": None,
            "dp_size": 1,
            "module": _SCHEDULER_MODULE,
            "nnodes": 1,
            "node_rank": 0,
            "pp_rank": 0,
            "pp_size": 1,
            "qualname": _SCHEDULER_QUALNAME,
            "start_method": "spawn",
            "trampoline_module": _TRAMPOLINE_MODULE,
            "trampoline_qualname": _TRAMPOLINE_QUALNAME,
        },
        "schema": _ACTIVATION_POLICY_SCHEMA,
        "transport": "stock-driver-cloexec-ready-frame",
    }
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _strict_int(
    value: object, *, field_name: str, minimum: int = 0, maximum: int | None = None
) -> int:
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        raise DiscoveryOverlayActivationError(
            f"discovery scheduler {field_name} is invalid"
        )
    return value


@dataclass(frozen=True)
class DiscoveryDriverOrigin:
    """Stable result of checking the driver against the installed distribution."""

    distribution: str
    version: str
    module: str

    def __post_init__(self) -> None:
        if (
            self.distribution != "sglang"
            or self.module != _DRIVER_MODULE
            or not isinstance(self.version, str)
            or _VERSION.fullmatch(self.version) is None
        ):
            raise DiscoveryOverlayActivationError(
                "discovery driver origin receipt is malformed"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "distribution": self.distribution,
            "module": self.module,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: object) -> "DiscoveryDriverOrigin":
        if not isinstance(value, Mapping) or set(value) != {
            "distribution", "module", "version",
        }:
            raise DiscoveryOverlayActivationError(
                "discovery driver origin fields are malformed"
            )
        return cls(
            distribution=value["distribution"],  # type: ignore[arg-type]
            version=value["version"],  # type: ignore[arg-type]
            module=value["module"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class DiscoverySchedulerMember:
    """One stock-driver-observed scheduler spawn, excluding process objects."""

    pid: int
    gpu_id: int
    tp_rank: int
    attn_cp_rank: int
    moe_dp_rank: int
    moe_ep_rank: int
    pp_rank: int
    dp_rank: int | None

    def __post_init__(self) -> None:
        for name in (
            "pid",
            "gpu_id",
            "tp_rank",
            "attn_cp_rank",
            "moe_dp_rank",
            "moe_ep_rank",
            "pp_rank",
        ):
            minimum = 1 if name == "pid" else 0
            object.__setattr__(
                self,
                name,
                _strict_int(getattr(self, name), field_name=name, minimum=minimum),
            )
        if self.dp_rank is not None:
            object.__setattr__(
                self,
                "dp_rank",
                _strict_int(self.dp_rank, field_name="dp_rank"),
            )

    def to_dict(self) -> dict[str, int | None]:
        return {
            "attn_cp_rank": self.attn_cp_rank,
            "dp_rank": self.dp_rank,
            "gpu_id": self.gpu_id,
            "moe_dp_rank": self.moe_dp_rank,
            "moe_ep_rank": self.moe_ep_rank,
            "pid": self.pid,
            "pp_rank": self.pp_rank,
            "tp_rank": self.tp_rank,
        }

    @classmethod
    def from_dict(cls, value: object) -> "DiscoverySchedulerMember":
        fields = {
            "attn_cp_rank", "dp_rank", "gpu_id", "moe_dp_rank",
            "moe_ep_rank", "pid", "pp_rank", "tp_rank",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise DiscoveryOverlayActivationError(
                "discovery scheduler member fields are malformed"
            )
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class DiscoveryActivationReceipt:
    """Candidate-inaccessible aggregate emitted by the stock engine driver."""

    schema: str
    overlay_identity_digest: str
    driver_pid: int
    driver_origin: DiscoveryDriverOrigin
    scheduler_target_module: str
    scheduler_target_qualname: str
    tp_size: int
    members: tuple[DiscoverySchedulerMember, ...]
    activation_policy_digest: str = field(default_factory=activation_policy_digest)

    def __post_init__(self) -> None:
        if self.schema != _ACTIVATION_SCHEMA:
            raise DiscoveryOverlayActivationError(
                "discovery activation receipt schema is unsupported"
            )
        if self.activation_policy_digest != activation_policy_digest():
            raise DiscoveryOverlayActivationError(
                "discovery activation receipt policy is unsupported"
            )
        if (
            _DIGEST.fullmatch(self.overlay_identity_digest) is None
            or self.overlay_identity_digest == "0" * 64
        ):
            raise DiscoveryOverlayActivationError(
                "discovery activation receipt identity is invalid"
            )
        _strict_int(self.driver_pid, field_name="driver_pid", minimum=1)
        _strict_int(
            self.tp_size, field_name="tp_size", minimum=1, maximum=_MAX_MEMBERS
        )
        if type(self.driver_origin) is not DiscoveryDriverOrigin:
            raise DiscoveryOverlayActivationError(
                "discovery activation driver origin is not typed"
            )
        if (
            self.scheduler_target_module != _SCHEDULER_MODULE
            or self.scheduler_target_qualname != _SCHEDULER_QUALNAME
        ):
            raise DiscoveryOverlayActivationError(
                "discovery activation names another scheduler target"
            )
        if (
            not isinstance(self.members, tuple)
            or len(self.members) != self.tp_size
            or any(type(row) is not DiscoverySchedulerMember for row in self.members)
        ):
            raise DiscoveryOverlayActivationError(
                "discovery activation member count or type is invalid"
            )
        ordered = tuple(sorted(self.members, key=lambda row: row.tp_rank))
        if self.members != ordered:
            raise DiscoveryOverlayActivationError(
                "discovery activation members are not in TP-rank order"
            )
        if (
            {row.tp_rank for row in ordered} != set(range(self.tp_size))
            or len({row.pid for row in ordered}) != self.tp_size
            or len({row.gpu_id for row in ordered}) != self.tp_size
            or self.driver_pid in {row.pid for row in ordered}
            or any(row.pp_rank != 0 or row.dp_rank is not None for row in ordered)
        ):
            raise DiscoveryOverlayActivationError(
                "discovery activation does not cover one TP-only scheduler set"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "activation_policy_digest": self.activation_policy_digest,
            "driver_origin": self.driver_origin.to_dict(),
            "driver_pid": self.driver_pid,
            "members": [row.to_dict() for row in self.members],
            "overlay_identity_digest": self.overlay_identity_digest,
            "scheduler_target_module": self.scheduler_target_module,
            "scheduler_target_qualname": self.scheduler_target_qualname,
            "schema": self.schema,
            "tp_size": self.tp_size,
        }

    @classmethod
    def from_dict(cls, value: object) -> "DiscoveryActivationReceipt":
        fields = {
            "activation_policy_digest", "driver_origin", "driver_pid", "members",
            "overlay_identity_digest", "scheduler_target_module",
            "scheduler_target_qualname", "schema", "tp_size",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise DiscoveryOverlayActivationError(
                "discovery activation receipt fields are malformed"
            )
        raw_members = value["members"]
        if (
            not isinstance(raw_members, list)
            or not 1 <= len(raw_members) <= _MAX_MEMBERS
        ):
            raise DiscoveryOverlayActivationError(
                "discovery activation receipt member array is invalid"
            )
        return cls(
            schema=value["schema"],  # type: ignore[arg-type]
            activation_policy_digest=value["activation_policy_digest"],  # type: ignore[arg-type]
            overlay_identity_digest=value["overlay_identity_digest"],  # type: ignore[arg-type]
            driver_pid=value["driver_pid"],  # type: ignore[arg-type]
            driver_origin=DiscoveryDriverOrigin.from_dict(value["driver_origin"]),
            scheduler_target_module=value["scheduler_target_module"],  # type: ignore[arg-type]
            scheduler_target_qualname=value["scheduler_target_qualname"],  # type: ignore[arg-type]
            tp_size=value["tp_size"],  # type: ignore[arg-type]
            members=tuple(
                DiscoverySchedulerMember.from_dict(row) for row in raw_members
            ),
        )


@dataclass(frozen=True)
class _ObservedScheduler:
    process: object
    member: DiscoverySchedulerMember


@dataclass(frozen=True)
class _PendingScheduler:
    gpu_id: int
    tp_rank: int
    attn_cp_rank: int
    moe_dp_rank: int
    moe_ep_rank: int
    pp_rank: int
    dp_rank: int | None


@dataclass
class _DriverActivationState:
    driver_pid: int
    overlay_identity_digest: str
    expected_members: int
    observed: list[_ObservedScheduler] = field(default_factory=list)
    closed: bool = False


_DRIVER_ACTIVATION: _DriverActivationState | None = None


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _required(environment: Mapping[str, str], key: str) -> str:
    value = str(environment.get(key, "")).strip()
    if not value:
        raise DiscoveryOverlayActivationError(
            f"discovery overlay bootstrap is missing {key}"
        )
    return value


def _required_digest(environment: Mapping[str, str], key: str) -> str:
    value = _required(environment, key)
    if _DIGEST.fullmatch(value) is None or value == "0" * 64:
        raise DiscoveryOverlayActivationError(
            f"discovery overlay bootstrap has an invalid {key}"
        )
    return value


def _required_pid(environment: Mapping[str, str], key: str) -> int:
    value = _required(environment, key)
    if not value.isascii() or not value.isdecimal():
        raise DiscoveryOverlayActivationError(
            f"discovery overlay bootstrap has an invalid {key}"
        )
    return _strict_int(int(value), field_name=key, minimum=1)


def _canonical_absolute(value: str, *, field: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or path != Path(os.path.normpath(path))
        or "\x00" in value
    ):
        raise DiscoveryOverlayActivationError(f"{field} is not a canonical absolute path")
    return path


def launch_environment(
    *,
    overlay_root: str | Path,
    expected_identity_digest: str,
    driver_pid: int | None = None,
) -> dict[str, str]:
    """Return the closed worker-owned environment for one discovery engine."""

    root = _canonical_absolute(str(overlay_root), field="overlay_root")
    if (
        _DIGEST.fullmatch(expected_identity_digest) is None
        or expected_identity_digest == "0" * 64
    ):
        raise DiscoveryOverlayActivationError("expected identity is not a SHA-256 digest")
    pid = os.getpid() if driver_pid is None else driver_pid
    if type(pid) is not int or pid < 1:
        raise DiscoveryOverlayActivationError("driver_pid must be a positive integer")
    result = {key: "" for key in DISCOVERY_ENVIRONMENT_KEYS}
    result.update({
        ARMED: "1",
        DRIVER_PID: str(pid),
        EXPECTED_IDENTITY: expected_identity_digest,
        OVERLAY_ROOT: str(root),
    })
    return result


def disabled_environment() -> dict[str, str]:
    """Explicitly clear every discovery marker for B, B-prime, and pristine T."""

    return {key: "" for key in DISCOVERY_ENVIRONMENT_KEYS}


def _target_identity(target: object) -> tuple[str, str]:
    while isinstance(target, functools.partial):
        target = target.func
    target = getattr(target, "__func__", target)
    return (
        str(getattr(target, "__module__", "")),
        str(getattr(target, "__qualname__", getattr(target, "__name__", ""))),
    )


def role_for_process_target(target: object) -> str | None:
    """Classify only the pinned SGLang scheduler process entry point."""

    module, qualname = _target_identity(target)
    if module == _SCHEDULER_MODULE and qualname == _SCHEDULER_QUALNAME:
        return _SCHEDULER_ROLE
    return None


def arm_driver_activation(
    *,
    expected_identity_digest: str,
    expected_members: int,
    driver_pid: int | None = None,
) -> None:
    """Start one stock-driver-owned scheduler observation window."""

    global _DRIVER_ACTIVATION
    if (
        _DIGEST.fullmatch(expected_identity_digest) is None
        or expected_identity_digest == "0" * 64
    ):
        raise DiscoveryOverlayActivationError(
            "expected driver activation identity is invalid"
        )
    members = _strict_int(
        expected_members,
        field_name="expected_members",
        minimum=1,
        maximum=_MAX_MEMBERS,
    )
    pid = os.getpid() if driver_pid is None else _strict_int(
        driver_pid, field_name="driver_pid", minimum=1
    )
    with _ENV_SPAWN_LOCK:
        if _DRIVER_ACTIVATION is not None:
            raise DiscoveryOverlayActivationError(
                "discovery driver activation is already armed"
            )
        _DRIVER_ACTIVATION = _DriverActivationState(
            driver_pid=pid,
            overlay_identity_digest=expected_identity_digest,
            expected_members=members,
        )


def clear_driver_activation() -> None:
    """Discard one driver observation window without producing authority."""

    global _DRIVER_ACTIVATION
    with _ENV_SPAWN_LOCK:
        _DRIVER_ACTIVATION = None


def _server_arg_int(server_args: object, name: str) -> int:
    return _strict_int(
        getattr(server_args, name, None), field_name=f"server_args.{name}"
    )


def _scheduler_member_before_start(
    process: object, state: _DriverActivationState
) -> _PendingScheduler:
    target = _target_identity(getattr(process, "_target", None))
    if target != (_SCHEDULER_MODULE, _SCHEDULER_QUALNAME):
        raise DiscoveryOverlayActivationError(
            "discovery scheduler target differs from the pinned entry point"
        )
    arguments = getattr(process, "_args", None)
    keywords = getattr(process, "_kwargs", None)
    if (
        not isinstance(arguments, tuple)
        or len(arguments) != 10
        or not isinstance(keywords, dict)
        or keywords
    ):
        raise DiscoveryOverlayActivationError(
            "pinned discovery scheduler signature changed"
        )
    server_args = arguments[0]
    if (
        _server_arg_int(server_args, "tp_size") != state.expected_members
        or _server_arg_int(server_args, "dp_size") != 1
        or _server_arg_int(server_args, "pp_size") != 1
        or _server_arg_int(server_args, "nnodes") != 1
        or _server_arg_int(server_args, "node_rank") != 0
    ):
        raise DiscoveryOverlayActivationError(
            "discovery activation supports only one-node DP1/PP1 TP engines"
        )
    gpu_id, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank, pp_rank = (
        _strict_int(arguments[index], field_name=name)
        for index, name in (
            (2, "gpu_id"),
            (3, "tp_rank"),
            (4, "attn_cp_rank"),
            (5, "moe_dp_rank"),
            (6, "moe_ep_rank"),
            (7, "pp_rank"),
        )
    )
    dp_rank = arguments[8]
    if dp_rank is not None:
        _strict_int(dp_rank, field_name="dp_rank")
    if pp_rank != 0 or dp_rank is not None:
        raise DiscoveryOverlayActivationError(
            "discovery scheduler member is outside the DP1/PP1 lane"
        )
    return _PendingScheduler(
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        attn_cp_rank=attn_cp_rank,
        moe_dp_rank=moe_dp_rank,
        moe_ep_rank=moe_ep_rank,
        pp_rank=pp_rank,
        dp_rank=dp_rank,
    )


def _scheduler_argument_blob(value: object, *, field: str) -> tuple[bytes, str]:
    try:
        raw = pickle.dumps(value, protocol=_SCHEDULER_PICKLE_PROTOCOL)
    except (
        pickle.PickleError,
        AttributeError,
        OSError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise DiscoveryOverlayActivationError(
            f"discovery scheduler {field} cannot be sealed: {exc}"
        ) from None
    if not raw or len(raw) > _MAX_SCHEDULER_ARGUMENT_BYTES:
        raise DiscoveryOverlayActivationError(
            f"discovery scheduler {field} pickle exceeds its hard bound"
        )
    return raw, hashlib.sha256(raw).hexdigest()


def _scheduler_argument_envelope(arguments: object) -> tuple[object, ...]:
    """Hide SGLang-owned arguments from ``spawn`` until overlay activation."""

    if not isinstance(arguments, tuple) or len(arguments) != 10:
        raise DiscoveryOverlayActivationError(
            "pinned discovery scheduler signature changed"
        )
    module = sys.modules.get(_SERVER_ARGS_MODULE)
    server_type = getattr(module, "ServerArgs", None)
    port_type = getattr(module, "PortArgs", None)
    server_args, port_args = arguments[:2]
    if (
        not isinstance(server_type, type)
        or not isinstance(port_type, type)
        or type(server_args) is not server_type
        or type(port_args) is not port_type
    ):
        raise DiscoveryOverlayActivationError(
            "discovery scheduler arguments are not exact pinned SGLang types"
        )
    if (
        not hasattr(server_args, "custom_sigquit_handler")
        or server_args.custom_sigquit_handler is not None
    ):
        raise DiscoveryOverlayActivationError(
            "discovery scheduler custom SIGQUIT handlers are unsupported"
        )
    server_blob, server_digest = _scheduler_argument_blob(
        server_args, field="ServerArgs"
    )
    port_blob, port_digest = _scheduler_argument_blob(port_args, field="PortArgs")
    return (
        _SCHEDULER_ARGUMENT_SCHEMA,
        server_blob,
        server_digest,
        port_blob,
        port_digest,
        *arguments[2:],
    )


def _open_scheduler_argument_blob(
    raw: object, digest: object, *, field: str, expected_type: type
) -> object:
    if (
        not isinstance(raw, bytes)
        or not raw
        or len(raw) > _MAX_SCHEDULER_ARGUMENT_BYTES
        or not isinstance(digest, str)
        or _DIGEST.fullmatch(digest) is None
        or hashlib.sha256(raw).hexdigest() != digest
    ):
        raise DiscoveryOverlayActivationError(
            f"discovery scheduler {field} envelope is malformed or changed"
        )
    try:
        value = pickle.loads(raw)
    except (
        pickle.PickleError,
        AttributeError,
        EOFError,
        ImportError,
        IndexError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise DiscoveryOverlayActivationError(
            f"discovery scheduler {field} cannot be reopened: {exc}"
        ) from None
    if type(value) is not expected_type:
        raise DiscoveryOverlayActivationError(
            f"discovery scheduler {field} did not reopen as the exact overlay type"
        )
    return value


def _open_scheduler_argument_envelope(
    arguments: tuple[object, ...],
) -> tuple[object, ...]:
    if len(arguments) != 13 or arguments[0] != _SCHEDULER_ARGUMENT_SCHEMA:
        raise DiscoveryOverlayActivationError(
            "discovery scheduler argument envelope schema or shape changed"
        )
    module = importlib.import_module("sglang.srt.server_args")
    server_type = getattr(module, "ServerArgs", None)
    port_type = getattr(module, "PortArgs", None)
    if not isinstance(server_type, type) or not isinstance(port_type, type):
        raise DiscoveryOverlayActivationError(
            "discovery overlay lacks exact scheduler argument classes"
        )
    server_args = _open_scheduler_argument_blob(
        arguments[1], arguments[2], field="ServerArgs", expected_type=server_type
    )
    port_args = _open_scheduler_argument_blob(
        arguments[3], arguments[4], field="PortArgs", expected_type=port_type
    )
    if (
        not hasattr(server_args, "custom_sigquit_handler")
        or server_args.custom_sigquit_handler is not None
    ):
        raise DiscoveryOverlayActivationError(
            "discovery overlay reopened a custom SIGQUIT handler"
        )
    return (server_args, port_args, *arguments[5:])


def _with_started_pid(
    member: _PendingScheduler, process: object
) -> DiscoverySchedulerMember:
    pid = _strict_int(getattr(process, "pid", None), field_name="pid", minimum=1)
    return DiscoverySchedulerMember(
        pid=pid,
        gpu_id=member.gpu_id,
        tp_rank=member.tp_rank,
        attn_cp_rank=member.attn_cp_rank,
        moe_dp_rank=member.moe_dp_rank,
        moe_ep_rank=member.moe_ep_rank,
        pp_rank=member.pp_rank,
        dp_rank=member.dp_rank,
    )


def require_stock_driver_origin(
    module: object,
    overlay_root: str | Path,
    *,
    expected_sglang_version: str,
    distribution: object | None = None,
    search_path: Sequence[str] | None = None,
) -> DiscoveryDriverOrigin:
    """Require the driver module to be the pinned installed distribution."""

    if (
        not isinstance(expected_sglang_version, str)
        or _VERSION.fullmatch(expected_sglang_version) is None
    ):
        raise DiscoveryOverlayActivationError(
            "expected SGLang version is invalid"
        )
    try:
        installed = (
            importlib.metadata.distribution("sglang")
            if distribution is None
            else distribution
        )
        installed_version = str(getattr(installed, "version"))
        requested_file = Path(
            installed.locate_file(_DRIVER_MODULE)  # type: ignore[attr-defined]
        )
        if requested_file.exists():
            expected_file = requested_file.resolve(strict=True)
        else:
            # Editable image installs keep metadata in site-packages and the
            # package under an image-owned source root. Resolve the already
            # imported top-level spec without executing another module.
            live_spec = importlib.util.find_spec("sglang")
            locations = (
                ()
                if live_spec is None
                else tuple(
                    Path(value).resolve(strict=True)
                    for value in (live_spec.submodule_search_locations or ())
                )
            )
            if (
                live_spec is None
                or live_spec.origin is None
                or len(locations) != 1
            ):
                raise DiscoveryOverlayActivationError(
                    "installed editable SGLang origin is not exact"
                )
            expected_file = Path(live_spec.origin).resolve(strict=True)
            if locations != (expected_file.parent,):
                raise DiscoveryOverlayActivationError(
                    "installed editable SGLang package root is ambiguous"
                )
        module_file = Path(getattr(module, "__file__", "")).resolve(strict=True)
        module_spec = getattr(module, "__spec__", None)
        spec_file = Path(getattr(module_spec, "origin", "")).resolve(strict=True)
    except (
        AttributeError,
        importlib.metadata.PackageNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise DiscoveryOverlayActivationError(
            f"cannot resolve the installed SGLang driver origin: {exc}"
        ) from None
    overlay = _canonical_absolute(str(overlay_root), field="overlay_root").resolve()
    overlay_site = overlay / "site"
    paths = sys.path if search_path is None else search_path
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise DiscoveryOverlayActivationError("driver search path is not a sequence")
    overlay_on_path = False
    for value in paths:
        if not isinstance(value, str):
            raise DiscoveryOverlayActivationError(
                "driver search path contains a non-string entry"
            )
        try:
            resolved = Path(value or os.curdir).resolve()
        except (OSError, RuntimeError):
            continue
        if resolved == overlay_site or overlay_site in resolved.parents:
            overlay_on_path = True
            break
    if (
        installed_version != expected_sglang_version
        or getattr(module, "__name__", None) != "sglang"
        or getattr(module_spec, "name", None) != "sglang"
        or module_file != expected_file
        or spec_file != expected_file
        or not expected_file.is_file()
        or module_file == overlay_site
        or overlay_site in module_file.parents
        or overlay_on_path
    ):
        raise DiscoveryOverlayActivationError(
            "engine driver did not import the exact pinned stock SGLang distribution"
        )
    return DiscoveryDriverOrigin("sglang", installed_version, _DRIVER_MODULE)


def _scheduler_overlay_entry(*args):
    """Activate the sealed overlay before importing the scheduler target.

    ``multiprocessing.spawn`` otherwise imports stock SGLang while unpickling
    both the target and its ``ServerArgs``/``PortArgs`` values, which is already
    too late to switch package roots.  The stock driver verifies and seals those
    values before substituting this validator-owned entry point.
    """

    install()
    expected = _required_digest(os.environ, EXPECTED_IDENTITY)
    try:
        module = importlib.import_module("sglang.srt.managers.scheduler")
        target = getattr(module, _SCHEDULER_QUALNAME)
    except (AttributeError, ImportError) as exc:
        raise DiscoveryOverlayActivationError(
            f"discovery scheduler target cannot import from the overlay: {exc}"
        ) from None
    if (
        _target_identity(target) != (_SCHEDULER_MODULE, _SCHEDULER_QUALNAME)
        or os.environ.get(ACTIVE_IDENTITY) != expected
    ):
        raise DiscoveryOverlayActivationError(
            "discovery scheduler trampoline did not activate the sealed overlay"
        )
    reopened = _open_scheduler_argument_envelope(args)
    return target(*reopened)


def install_process_role_hook() -> None:
    """Mark only exact scheduler spawn children while an overlay is armed."""

    base = multiprocessing.process.BaseProcess
    current = base.start
    if getattr(current, "__optima_discovery_role_hook__", False):
        return
    original = current

    def start(process, *args, **kwargs):
        if not _truthy(os.environ.get(ARMED)):
            return original(process, *args, **kwargs)
        role = role_for_process_target(getattr(process, "_target", None))
        # Non-scheduler children also take the lock so they cannot inherit the
        # short-lived marker exported for a concurrent scheduler spawn.
        with _ENV_SPAWN_LOCK:
            if role is None:
                return original(process, *args, **kwargs)
            state = _DRIVER_ACTIVATION
            if (
                state is None
                or state.closed
                or state.driver_pid != os.getpid()
                or len(state.observed) >= state.expected_members
                or _required_digest(os.environ, EXPECTED_IDENTITY)
                != state.overlay_identity_digest
                or _required_pid(os.environ, DRIVER_PID) != state.driver_pid
            ):
                raise DiscoveryOverlayActivationError(
                    "scheduler spawn is outside the armed stock-driver window"
                )
            method = getattr(process, "_start_method", None)
            if method is None:
                method = multiprocessing.get_start_method(allow_none=False)
            if method != "spawn":
                raise DiscoveryOverlayActivationError(
                    "discovery overlays require a spawn scheduler; "
                    f"observed {method!r}"
                )
            pending = _scheduler_member_before_start(process, state)
            original_arguments = getattr(process, "_args", None)
            sealed_arguments = _scheduler_argument_envelope(original_arguments)
            if (
                any(
                    row.member.tp_rank == pending.tp_rank
                    for row in state.observed
                )
                or any(
                    row.member.gpu_id == pending.gpu_id
                    for row in state.observed
                )
            ):
                raise DiscoveryOverlayActivationError(
                    "stock driver requested a duplicate discovery scheduler member"
                )
            saved_role = os.environ.get(PROCESS_ROLE)
            saved_parent = os.environ.get(ROLE_PARENT_PID)
            original_target = getattr(process, "_target", None)
            os.environ[PROCESS_ROLE] = role
            os.environ[ROLE_PARENT_PID] = str(os.getpid())
            process._target = _scheduler_overlay_entry
            process._args = sealed_arguments
            try:
                result = original(process, *args, **kwargs)
                member = _with_started_pid(pending, process)
                if any(row.member.pid == member.pid for row in state.observed):
                    raise DiscoveryOverlayActivationError(
                        "stock driver observed a duplicate discovery scheduler PID"
                    )
                state.observed.append(_ObservedScheduler(process, member))
                return result
            finally:
                if saved_role is None:
                    os.environ.pop(PROCESS_ROLE, None)
                else:
                    os.environ[PROCESS_ROLE] = saved_role
                if saved_parent is None:
                    os.environ.pop(ROLE_PARENT_PID, None)
                else:
                    os.environ[ROLE_PARENT_PID] = saved_parent
                process._args = original_arguments
                process._target = original_target

    start.__optima_discovery_role_hook__ = True
    start.__optima_original_start__ = original
    base.start = start


def activate_scheduler_overlay(
    *,
    environment: MutableMapping[str, str] | None = None,
    pid: int | None = None,
    parent_pid: int | None = None,
    modules: Mapping[str, object] | None = None,
    sys_path: list[str] | None = None,
    reader: Callable[..., object] | None = None,
    read_only_check: Callable[[Path], bool] | None = None,
    defer_to_import: bool = False,
) -> tuple[Path, str] | None:
    """Validate and arm the overlay only inside one marked scheduler child."""

    environ = os.environ if environment is None else environment
    if not _truthy(environ.get(ARMED)):
        return None
    role = str(environ.get(PROCESS_ROLE, "")).strip()
    if not role:
        return None
    if role != _SCHEDULER_ROLE:
        raise DiscoveryOverlayActivationError(
            f"unrecognized discovery process role: {role!r}"
        )

    actual_pid = os.getpid() if pid is None else pid
    actual_parent = os.getppid() if parent_pid is None else parent_pid
    try:
        driver = int(_required(environ, DRIVER_PID))
        marked_parent = int(_required(environ, ROLE_PARENT_PID))
    except ValueError as exc:
        raise DiscoveryOverlayActivationError(
            "discovery overlay PID markers are malformed"
        ) from exc
    if actual_pid == driver:
        raise DiscoveryOverlayActivationError(
            "refusing discovery overlay activation in the engine driver"
        )
    if actual_parent != marked_parent:
        raise DiscoveryOverlayActivationError(
            "discovery scheduler parent marker does not match the live parent"
        )
    loaded = sys.modules if modules is None else modules
    if any(name == "sglang" or name.startswith("sglang.") for name in loaded):
        raise DiscoveryOverlayActivationError(
            "discovery activation occurred after an SGLang import"
        )

    root = _canonical_absolute(_required(environ, OVERLAY_ROOT), field="overlay root")
    expected = _required_digest(environ, EXPECTED_IDENTITY)
    if reader is None:
        from optima.discovery import reopen_discovery_overlay
        from optima.eval.native_artifact import reopen_native_artifact

        publication_root = _canonical_absolute(
            _required(environ, "OPTIMA_NATIVE_ARTIFACT_ROOT"),
            field="native artifact root",
        )
        publication = reopen_native_artifact(
            publication_root,
            expected_build_spec_digest=_required_digest(
                environ, "OPTIMA_NATIVE_BUILD_SPEC_DIGEST"
            ),
            expected_publication_digest=_required_digest(
                environ, "OPTIMA_NATIVE_ARTIFACT_PUBLICATION_DIGEST"
            ),
            expected_owner_uid=None,
        )
        kwargs = {"expected_identity_digest": expected, "require_read_only": True}
        if read_only_check is not None:
            kwargs["read_only_check"] = read_only_check
        overlay = reopen_discovery_overlay(publication, **kwargs)
    else:
        kwargs = {"expected_identity_digest": expected, "require_read_only": True}
        if read_only_check is not None:
            kwargs["read_only_check"] = read_only_check
        overlay = reader(root, **kwargs)
    identity = getattr(getattr(overlay, "identity", None), "digest", None)
    overlay_root = Path(getattr(overlay, "root", root)).resolve()
    if identity != expected or overlay_root != root.resolve():
        raise DiscoveryOverlayActivationError(
            "discovery overlay identity or root differs from the worker binding"
        )
    site = overlay_root / "site"
    package = site / "sglang"
    if site.is_symlink() or package.is_symlink() or not package.is_dir():
        raise DiscoveryOverlayActivationError(
            "validated discovery overlay has no safe SGLang package"
        )
    if not defer_to_import:
        paths = sys.path if sys_path is None else sys_path
        site_text = str(site)
        if site_text in paths:
            paths.remove(site_text)
        paths.insert(0, site_text)
        importlib.invalidate_caches()
        environ[ACTIVE_IDENTITY] = expected
    return site, expected


class _OverlayOriginLoader(importlib.abc.Loader):
    def __init__(
        self,
        delegate: object,
        expected_origin: Path,
        complete: Callable[[Path], None],
    ):
        self.delegate = delegate
        self.expected_origin = expected_origin
        self.complete = complete

    def create_module(self, spec):
        create = getattr(self.delegate, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module) -> None:
        try:
            self.delegate.exec_module(module)
            actual = Path(module.__file__).resolve(strict=True)
            if actual != self.expected_origin:
                raise ImportError("loaded SGLang escaped the discovery overlay")
            self.complete(actual)
        except BaseException as exc:
            _install_failure_guard(f"{type(exc).__name__}: {exc}"[:2048])
            raise

    def __getattr__(self, name: str):
        return getattr(self.delegate, name)


class _SchedulerOverlayFinder(importlib.abc.MetaPathFinder):
    """Force the overlay root after multiprocessing restores the parent path."""

    def __init__(
        self,
        site: Path,
        expected_identity: str,
        environment: MutableMapping[str, str] | None = None,
    ):
        self.site = site.resolve()
        self.expected_identity = expected_identity
        self.environment = os.environ if environment is None else environment

    def _complete(self, module_origin: Path) -> None:
        # Diagnostic only.  Crown authority is the stock driver's in-memory
        # scheduler ledger, transported later over the worker's CLOEXEC pipe.
        self.environment[ACTIVE_IDENTITY] = self.expected_identity

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "sglang":
            if fullname.startswith("sglang."):
                raise ImportError(
                    "SGLang submodule import preceded discovery package activation"
                )
            return None
        package = self.site / "sglang"
        spec = importlib.machinery.PathFinder.find_spec(fullname, [str(self.site)])
        origin = getattr(spec, "origin", None) if spec is not None else None
        try:
            resolved_origin = Path(origin).resolve(strict=True)
        except (TypeError, OSError, RuntimeError):
            resolved_origin = None
        expected_origin = (package / "__init__.py").resolve(strict=True)
        locations = (
            tuple(
                Path(value).resolve()
                for value in (getattr(spec, "submodule_search_locations", None) or ())
            )
            if spec is not None
            else ()
        )
        if (
            spec is None
            or spec.loader is None
            or resolved_origin != expected_origin
            or locations != (package.resolve(strict=True),)
        ):
            raise ImportError("discovery overlay did not resolve the exact SGLang root")
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        site_text = str(self.site)
        if site_text in sys.path:
            sys.path.remove(site_text)
        sys.path.insert(0, site_text)
        inherited = [
            value
            for value in self.environment.get("PYTHONPATH", "").split(os.pathsep)
            if value and value != site_text
        ]
        self.environment["PYTHONPATH"] = os.pathsep.join([site_text, *inherited])
        importlib.invalidate_caches()
        spec.loader = _OverlayOriginLoader(spec.loader, expected_origin, self._complete)
        return spec


class _BlockSglangImport(importlib.abc.MetaPathFinder):
    def __init__(self, reason: str):
        self.reason = reason

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sglang" or fullname.startswith("sglang."):
            raise ImportError(
                "discovery overlay activation failed closed: " + self.reason
            )
        return None


def _install_failure_guard(reason: str) -> None:
    if not any(isinstance(finder, _BlockSglangImport) for finder in sys.meta_path):
        sys.meta_path.insert(0, _BlockSglangImport(reason))
    os.environ[ACTIVATION_FAILED] = reason[:2048]


def install() -> None:
    """Install spawn-role plumbing and scheduler import activation."""

    install_process_role_hook()
    if not _truthy(os.environ.get(ARMED)) or not os.environ.get(PROCESS_ROLE):
        return
    try:
        activated = activate_scheduler_overlay(defer_to_import=True)
        if activated is not None:
            site, expected = activated
            sys.meta_path.insert(0, _SchedulerOverlayFinder(site, expected))
    except Exception as exc:
        _install_failure_guard(f"{type(exc).__name__}: {exc}")
    finally:
        # Role markers apply to exactly one spawn child.  Retaining them would
        # misclassify later helpers created by that scheduler.
        os.environ.pop(PROCESS_ROLE, None)
        os.environ.pop(ROLE_PARENT_PID, None)


def require_driver_activation(
    sglang_module: object,
    overlay_root: str | Path,
    *,
    expected_identity_digest: str,
    expected_members: int,
    expected_sglang_version: str,
    distribution: object | None = None,
    search_path: Sequence[str] | None = None,
    driver_pid: int | None = None,
) -> DiscoveryActivationReceipt:
    """Close the driver window and return one TP-complete typed receipt."""

    global _DRIVER_ACTIVATION
    identity = _required_digest(
        {EXPECTED_IDENTITY: expected_identity_digest}, EXPECTED_IDENTITY
    )
    members = _strict_int(
        expected_members,
        field_name="expected_members",
        minimum=1,
        maximum=_MAX_MEMBERS,
    )
    pid = os.getpid() if driver_pid is None else _strict_int(
        driver_pid, field_name="driver_pid", minimum=1
    )
    with _ENV_SPAWN_LOCK:
        state = _DRIVER_ACTIVATION
        try:
            environment_root = _canonical_absolute(
                _required(os.environ, OVERLAY_ROOT), field="overlay_root"
            ).resolve()
            expected_root = _canonical_absolute(
                str(overlay_root), field="overlay_root"
            ).resolve()
            environment_matches = (
                _truthy(os.environ.get(ARMED))
                and _required_digest(os.environ, EXPECTED_IDENTITY) == identity
                and _required_pid(os.environ, DRIVER_PID) == pid
                and environment_root == expected_root
            )
        except DiscoveryOverlayActivationError:
            environment_matches = False
        if (
            state is None
            or state.closed
            or state.driver_pid != pid
            or state.overlay_identity_digest != identity
            or state.expected_members != members
            or not environment_matches
        ):
            raise DiscoveryOverlayActivationError(
                "discovery driver activation window differs from the requested proof"
            )
        if len(state.observed) != members:
            raise DiscoveryOverlayActivationError(
                "discovery scheduler activation is incomplete "
                f"({len(state.observed)}/{members})"
            )
        typed_members: list[DiscoverySchedulerMember] = []
        for observed in state.observed:
            process = observed.process
            try:
                alive = process.is_alive()
                exitcode = process.exitcode
            except (AssertionError, AttributeError, OSError, ValueError) as exc:
                raise DiscoveryOverlayActivationError(
                    f"cannot prove a discovery scheduler is live: {exc}"
                ) from None
            if alive is not True or exitcode is not None:
                raise DiscoveryOverlayActivationError(
                    "discovery scheduler exited before the engine ready boundary"
                )
            typed_members.append(observed.member)
        origin = require_stock_driver_origin(
            sglang_module,
            overlay_root,
            expected_sglang_version=expected_sglang_version,
            distribution=distribution,
            search_path=search_path,
        )
        receipt = DiscoveryActivationReceipt(
            schema=_ACTIVATION_SCHEMA,
            overlay_identity_digest=identity,
            driver_pid=pid,
            driver_origin=origin,
            scheduler_target_module=_SCHEDULER_MODULE,
            scheduler_target_qualname=_SCHEDULER_QUALNAME,
            tp_size=members,
            members=tuple(sorted(typed_members, key=lambda row: row.tp_rank)),
        )
        state.closed = True
        return receipt


__all__ = [
    "ACTIVE_IDENTITY",
    "ACTIVATION_FAILED",
    "ARMED",
    "DISCOVERY_ENVIRONMENT_KEYS",
    "DiscoveryActivationReceipt",
    "DiscoveryDriverOrigin",
    "DiscoveryOverlayActivationError",
    "DiscoverySchedulerMember",
    "activation_policy_digest",
    "arm_driver_activation",
    "clear_driver_activation",
    "disabled_environment",
    "install",
    "install_process_role_hook",
    "launch_environment",
    "require_driver_activation",
    "require_stock_driver_origin",
    "role_for_process_target",
]
