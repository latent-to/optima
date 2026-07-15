from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest

from optima.artifact_provider import CUTE_CUBIN_PROVIDER_ID
from optima.eval.native_compile_profile import (
    PROFILE_SCHEMA,
    NativeCompileProfileError,
    NativeCuTeCompileProfile,
)
from optima.stack_identity import canonical_json_bytes


_CLUSTER_1 = "max_active_clusters.cluster_size_1"
_CLUSTER_4 = "max_active_clusters.cluster_size_4"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _profile(**changes: object) -> NativeCuTeCompileProfile:
    values: dict[str, object] = {
        "logical_architecture": "sm103",
        "compiler_architecture": "sm_103a",
        "image_digest": _digest("image"),
        "platform_digest": _digest("platform"),
        "worker_distribution_digest": _digest("worker"),
        "logical_hardware_digest": _digest("logical hardware"),
        "device_policy_digest": _digest("device policy"),
        "topology_digest": _digest("topology"),
        "visible_gpu_count": 8,
        "tp_size": 4,
        "ep_size": 1,
        "dp_size": 2,
        # Deliberately non-canonical input order.
        "constants": {_CLUSTER_4: 2, _CLUSTER_1: 8},
        "measurement_digest": _digest("measurement"),
    }
    values.update(changes)
    return NativeCuTeCompileProfile(**values)  # type: ignore[arg-type]


def _launch_facts(profile: NativeCuTeCompileProfile) -> dict[str, object]:
    return {
        "image_digest": profile.image_digest,
        "platform_digest": profile.platform_digest,
        "worker_distribution_digest": profile.worker_distribution_digest,
        "logical_hardware_digest": profile.logical_hardware_digest,
        "logical_architecture": profile.logical_architecture,
        "device_policy_digest": profile.device_policy_digest,
        "topology_digest": profile.topology_digest,
        "visible_gpu_count": profile.visible_gpu_count,
        "tp_size": profile.tp_size,
        "ep_size": profile.ep_size,
        "dp_size": profile.dp_size,
    }


def test_profile_is_canonical_round_trippable_and_read_only() -> None:
    profile = _profile()
    expected_constants = ((_CLUSTER_1, 8), (_CLUSTER_4, 2))

    assert profile.schema == PROFILE_SCHEMA
    assert profile.provider == CUTE_CUBIN_PROVIDER_ID
    assert profile.constants == expected_constants
    assert profile.to_dict()["constants"] == dict(expected_constants)
    assert (
        profile.canonical_bytes
        == canonical_json_bytes(profile.to_dict()) + b"\n"
    )
    assert json.loads(profile.canonical_bytes) == profile.to_dict()
    assert NativeCuTeCompileProfile.from_dict(profile.to_dict()) == profile
    assert NativeCuTeCompileProfile.from_dict(
        json.loads(profile.canonical_bytes)
    ).digest == profile.digest
    assert dict(profile.values) == dict(expected_constants)
    assert profile.require_int(_CLUSTER_1) == 8

    with pytest.raises(TypeError):
        profile.values[_CLUSTER_1] = 9  # type: ignore[index]

    reordered = replace(
        profile,
        constants={_CLUSTER_1: 8, _CLUSTER_4: 2},
    )
    assert reordered == profile
    assert reordered.digest == profile.digest


def test_profile_digest_binds_measurement_architecture_and_values() -> None:
    profile = _profile()
    variants = (
        replace(profile, measurement_digest=_digest("other measurement")),
        replace(profile, compiler_architecture="sm_103"),
        replace(profile, constants={_CLUSTER_1: 7, _CLUSTER_4: 2}),
        replace(profile, topology_digest=_digest("other topology")),
    )
    for variant in variants:
        assert variant.digest != profile.digest


def test_profile_is_bound_to_the_registered_device_only_cubin_provider() -> None:
    profile = _profile()

    assert profile.provider == CUTE_CUBIN_PROVIDER_ID
    assert profile.require_int(_CLUSTER_1) == 8
    assert NativeCuTeCompileProfile.from_dict(profile.to_dict()) == profile
    with pytest.raises(NativeCompileProfileError, match="provider is unregistered"):
        _profile(provider="cutlass.cute.object.v1")


@pytest.mark.parametrize(
    "changes,message",
    (
        ({"schema": 1}, "schema mismatch"),
        ({"provider": 1}, "provider is unregistered"),
        ({"logical_architecture": "SM103"}, "logical_architecture"),
        ({"compiler_architecture": "sm103"}, "compiler_architecture"),
        ({"compiler_architecture": "sm_120"}, "architecture family"),
        ({"image_digest": "0" * 64}, "all-zero"),
        ({"visible_gpu_count": True}, "visible_gpu_count"),
        ({"tp_size": 9}, "tp_size cannot exceed"),
        ({"constants": {_CLUSTER_1: True}}, "hard bound"),
        ({"constants": {_CLUSTER_1: 1_048_577}}, "hard bound"),
        ({"constants": {"miner.chosen_value": 1}}, "not registered"),
        ({"constants": [(_CLUSTER_1, 8)]}, "wrong type"),
        (
            {"constants": ((_CLUSTER_1, 8), (_CLUSTER_1, 7))},
            "duplicates",
        ),
    ),
)
def test_profile_constructor_rejects_malformed_authority(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(NativeCompileProfileError, match=message):
        _profile(**changes)


def test_profile_parser_requires_closed_outer_and_degree_schemas() -> None:
    profile = _profile()

    extra = profile.to_dict()
    extra["miner_value"] = 7
    with pytest.raises(NativeCompileProfileError, match="fields mismatch"):
        NativeCuTeCompileProfile.from_dict(extra)

    missing = profile.to_dict()
    del missing["measurement_digest"]
    with pytest.raises(NativeCompileProfileError, match="fields mismatch"):
        NativeCuTeCompileProfile.from_dict(missing)

    extra_degree = copy.deepcopy(profile.to_dict())
    extra_degree["degrees"]["pipeline_stages"] = 7  # type: ignore[index]
    with pytest.raises(NativeCompileProfileError, match="degrees mismatch"):
        NativeCuTeCompileProfile.from_dict(extra_degree)

    missing_degree = copy.deepcopy(profile.to_dict())
    del missing_degree["degrees"]["tp_size"]  # type: ignore[index]
    with pytest.raises(NativeCompileProfileError, match="degrees mismatch"):
        NativeCuTeCompileProfile.from_dict(missing_degree)


def test_require_int_rejects_missing_unknown_and_non_string_inputs() -> None:
    profile = _profile()

    with pytest.raises(NativeCompileProfileError, match="does not provide"):
        profile.require_int("max_active_clusters.cluster_size_2")
    with pytest.raises(NativeCompileProfileError, match="not registered"):
        profile.require_int("miner.chosen_value")
    with pytest.raises(NativeCompileProfileError, match="not registered"):
        profile.require_int([])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field,value",
    (
        ("image_digest", _digest("other image")),
        ("platform_digest", _digest("other platform")),
        ("worker_distribution_digest", _digest("other worker")),
        ("logical_hardware_digest", _digest("other logical hardware")),
        ("logical_architecture", "sm120"),
        ("device_policy_digest", _digest("other device policy")),
        ("topology_digest", _digest("other topology")),
        ("visible_gpu_count", 16),
        ("tp_size", 8),
        ("ep_size", 2),
        ("dp_size", 1),
    ),
)
def test_profile_validates_every_launch_authority_field(
    field: str, value: object
) -> None:
    profile = _profile()
    facts = _launch_facts(profile)
    profile.validate_launch(**facts)  # type: ignore[arg-type]

    facts[field] = value
    with pytest.raises(NativeCompileProfileError, match=field):
        profile.validate_launch(**facts)  # type: ignore[arg-type]
