"""Seam-activation receipts — the anti-phantom-pass gate (optima/receipts.py).

Pins the failure mode hit for real on 2026-07-07: a candidate engine that comes up
WITHOUT the seam (missing .pth bootstrap / bundle load fallback) produced
bit-identical logits, KL exactly 0.0, and a PASS verdict. The eval driver must
demand positive evidence from the ranks, and the diagnosis must distinguish
"no bootstrap at all" from "bundle load fell back to baseline".
"""

from __future__ import annotations

import os

import pytest

from optima import receipts
from optima.registry import Eligibility, KernelImpl, KernelRegistry


@pytest.fixture()
def receipt_dir(tmp_path, monkeypatch):
    rdir = tmp_path / "receipts"
    monkeypatch.setenv("OPTIMA_SEAM_RECEIPT_DIR", str(rdir))
    monkeypatch.setattr(receipts, "_ONCE", set())
    return rdir


def test_no_env_is_a_silent_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("OPTIMA_SEAM_RECEIPT_DIR", raising=False)
    receipts.write("active", {"bundle": "x"})  # must not raise, must not create files
    receipts.completed("norm.rmsnorm")
    assert list(tmp_path.iterdir()) == []


def test_write_and_collect_roundtrip(receipt_dir):
    receipts.write("active", {"bundle": "b", "slots": ["s"]})
    receipts.write("fired", {"slot": "collective.ar_residual_rmsnorm"},
                   tag="collective.ar_residual_rmsnorm")
    active = receipts.collect(receipt_dir, "active")
    assert active[0]["bundle"] == "b" and active[0]["slots"] == ["s"]
    assert active[0]["pid"] == os.getpid()
    fired = receipts.collect(receipt_dir, "fired")
    assert fired[0]["slot"] == "collective.ar_residual_rmsnorm"
    assert {"pid", "rank", "world_size"} <= fired[0].keys()
    # tag is sanitized into the filename; pid keeps concurrent ranks from colliding
    names = [p.name for p in receipt_dir.iterdir()]
    assert any(n.startswith("fired.collective.ar_residual_rmsnorm") for n in names)
    assert all(str(os.getpid()) in n for n in names)


def test_require_passes_with_receipt(receipt_dir):
    receipts.write("active", {"bundle": "b"})
    got = receipts.require(receipt_dir, "active", context="test")
    assert got and got[0]["bundle"] == "b"


def test_require_diagnoses_missing_bootstrap(receipt_dir):
    receipt_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="WITHOUT the miner kernel"):
        receipts.require(receipt_dir, "active", context="test")


def test_require_diagnoses_bundle_fallback(receipt_dir):
    receipts.write("load_failed", {"bundle": "b", "reason": "exception during load"})
    with pytest.raises(RuntimeError, match="FELL BACK to baseline"):
        receipts.require(receipt_dir, "active", context="test")


def test_registry_lookup_writes_fired_once(receipt_dir, monkeypatch):
    # The fired guard is process-global (one receipt per slot per process); isolate it
    # so earlier suite tests that exercised lookup() can't mask the write.
    monkeypatch.setattr("optima.registry._FIRED_SLOTS", set())
    reg = KernelRegistry()
    reg.register(KernelImpl(slot="activation.silu_and_mul", bundle_id="t",
                            entry=lambda *a: None, eligibility=Eligibility()))
    reg.enable()
    for _ in range(3):  # repeated lookups -> exactly one receipt (per-process guard)
        assert reg.lookup("activation.silu_and_mul", dtype_name="bfloat16",
                          last_dim=128, arch=None) is not None
    fired = receipts.collect(receipt_dir, "fired")
    assert len(fired) == 1 and fired[0]["slot"] == "activation.silu_and_mul"


def test_registry_miss_writes_nothing(receipt_dir, monkeypatch):
    monkeypatch.setattr("optima.registry._FIRED_SLOTS", set())
    reg = KernelRegistry()
    reg.register(KernelImpl(slot="norm.rmsnorm", bundle_id="t", entry=lambda *a: None,
                            eligibility=Eligibility(dtypes=frozenset({"float16"}))))
    reg.enable()
    # Ineligible (dtype mismatch) -> no selection -> no fired receipt.
    assert reg.lookup("norm.rmsnorm", dtype_name="bfloat16", last_dim=128, arch=None) is None
    assert receipts.collect(receipt_dir, "fired") == []


def test_no_env_does_not_consume_execution_once_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(receipts, "_ONCE", set())
    monkeypatch.delenv("OPTIMA_SEAM_RECEIPT_DIR", raising=False)
    receipts.completed("norm.rmsnorm")
    monkeypatch.setenv("OPTIMA_SEAM_RECEIPT_DIR", str(tmp_path))
    receipts.completed("norm.rmsnorm")
    assert len(receipts.collect(tmp_path, "completed")) == 1


def test_execution_once_guard_is_scoped_to_resolved_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(receipts, "_ONCE", set())
    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setenv("OPTIMA_SEAM_RECEIPT_DIR", str(first))
    receipts.completed("norm.rmsnorm")
    receipts.completed("norm.rmsnorm")
    monkeypatch.setenv(
        "OPTIMA_SEAM_RECEIPT_DIR", f"{first}/../{first.name}"
    )
    receipts.completed("norm.rmsnorm")
    monkeypatch.setenv("OPTIMA_SEAM_RECEIPT_DIR", str(second))
    receipts.completed("norm.rmsnorm")
    assert len(receipts.collect(first, "completed")) == 1
    assert len(receipts.collect(second, "completed")) == 1


def test_failed_execution_receipt_write_does_not_consume_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(receipts, "_ONCE", set())
    monkeypatch.setenv("OPTIMA_SEAM_RECEIPT_DIR", str(tmp_path))
    outcomes = iter((False, True))
    calls = []

    def fake_write(*_args, **_kwargs):
        calls.append(1)
        return next(outcomes)

    monkeypatch.setattr(receipts, "_write_to", fake_write)
    receipts.completed("slot.a")
    receipts.completed("slot.a")
    receipts.completed("slot.a")
    assert len(calls) == 2


def test_detected_identity_overrides_payload(receipt_dir, monkeypatch):
    monkeypatch.setattr(
        receipts, "identity", lambda: {"pid": 7, "rank": 1, "world_size": 2}
    )
    receipts.write(
        "completed",
        {"slot": "slot.a", "pid": 999, "rank": 999, "world_size": 999},
    )
    got = receipts.collect(receipt_dir, "completed")[0]
    assert (got["pid"], got["rank"], got["world_size"]) == (7, 1, 2)


def test_completed_and_fallback_are_independently_once(receipt_dir):
    for _ in range(3):
        receipts.completed("norm.rmsnorm")
        receipts.fallback("norm.rmsnorm", RuntimeError("candidate path failed"))
    completed = receipts.collect(receipt_dir, "completed")
    fallback = receipts.collect(receipt_dir, "fallback")
    assert len(completed) == len(fallback) == 1
    assert fallback[0]["error_type"] == "RuntimeError"


@pytest.mark.parametrize(
    ("environment", "expected_phase"),
    (
        ({}, "all"),
        (
            {
                "OPTIMA_ENGINE_WORKER": "1",
                "OPTIMA_PREBUILT_ARTIFACTS": "1",
            },
            "load",
        ),
    ),
)
def test_scheduler_bundle_rebuild_phase_matches_launch_authority(
    monkeypatch, environment, expected_phase
):
    """Production is load-only; explicit direct eval reuses its dev cache."""
    from optima import manifest, rebuild, sandbox
    from optima.registry import REGISTRY
    from optima.seam import _load_bundle_into_registry

    class EmptyManifest:
        ops = ()

    class CleanTree:
        ok = True
        violations = ()

    calls = []
    monkeypatch.setattr(manifest, "load_manifest", lambda _bundle: EmptyManifest())
    monkeypatch.setattr(
        manifest, "all_declared_cuda_sources", lambda _bundle, _manifest: ()
    )
    monkeypatch.setattr(
        manifest, "all_declared_dep_patches", lambda _bundle, _manifest: ()
    )
    monkeypatch.setattr(sandbox, "scan_tree", lambda *_args, **_kwargs: CleanTree())
    monkeypatch.setattr(
        rebuild,
        "apply_rebuild_plan",
        lambda bundle, *, phase: calls.append((bundle, phase)),
    )
    for name in ("OPTIMA_ENGINE_WORKER", "OPTIMA_PREBUILT_ARTIFACTS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    REGISTRY.clear()
    try:
        _load_bundle_into_registry("/sealed/candidate-tree")
    finally:
        REGISTRY.clear()

    assert calls == [("/sealed/candidate-tree", expected_phase)]


@pytest.mark.parametrize(
    "environment",
    (
        {"OPTIMA_ENGINE_WORKER": "1"},
        {"OPTIMA_PREBUILT_ARTIFACTS": "1"},
    ),
)
def test_scheduler_rejects_partial_native_artifact_authority(
    monkeypatch, environment
):
    from optima import manifest, sandbox
    from optima.registry import REGISTRY
    from optima.seam import _load_bundle_into_registry

    class EmptyManifest:
        ops = ()

    class CleanTree:
        ok = True
        violations = ()

    monkeypatch.setattr(manifest, "load_manifest", lambda _bundle: EmptyManifest())
    monkeypatch.setattr(manifest, "all_declared_cuda_sources", lambda *_args: ())
    monkeypatch.setattr(manifest, "all_declared_dep_patches", lambda *_args: ())
    monkeypatch.setattr(sandbox, "scan_tree", lambda *_args, **_kwargs: CleanTree())
    for name in ("OPTIMA_ENGINE_WORKER", "OPTIMA_PREBUILT_ARTIFACTS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    REGISTRY.clear()
    try:
        with pytest.raises(RuntimeError, match="incomplete native-artifact authority"):
            _load_bundle_into_registry("/sealed/candidate-tree")
    finally:
        REGISTRY.clear()


def test_unprintable_fallback_error_never_breaks_receipt(receipt_dir):
    class BadString(RuntimeError):
        def __str__(self):
            raise RuntimeError("format failed")

    receipts.fallback("slot.a", BadString())
    got = receipts.collect(receipt_dir, "fallback")[0]
    assert got["error"] == "<unprintable exception>"


@pytest.mark.parametrize("payload", ("{", "[]"))
def test_strict_collection_rejects_malformed_receipts(receipt_dir, payload):
    receipt_dir.mkdir(parents=True, exist_ok=True)
    (receipt_dir / "completed.bad.json").write_text(payload)
    with pytest.raises(receipts.ReceiptFormatError):
        receipts.collect(receipt_dir, "completed")


def _active_members(count=2):
    return [
        {"pid": 10 + rank, "rank": rank, "world_size": count, "slots": ["a", "b"]}
        for rank in range(count)
    ]


def _completed_members(count=2):
    return [
        {"slot": slot, "pid": 10 + rank, "rank": rank, "world_size": count}
        for rank in range(count)
        for slot in ("a", "b")
    ]


def test_completed_gate_requires_exact_slot_member_cross_product():
    active = _active_members()
    completed = _completed_members()
    ok, desc = receipts.completed_gate(
        completed,
        expected_slots=("a", "b"),
        member_receipts=active,
        expected_member_count=2,
    )
    assert ok and "4/4" in desc

    ok, desc = receipts.completed_gate(
        completed[:-1],
        expected_slots=("a", "b"),
        member_receipts=active,
        expected_member_count=2,
    )
    assert not ok and "pid:11" in desc and "slot" in desc


@pytest.mark.parametrize("active", (_active_members(1), _active_members(3)))
def test_completed_gate_rejects_wrong_active_member_count(active):
    completed = [
        {"slot": "a", "pid": row["pid"], "rank": row["rank"],
         "world_size": row["world_size"]}
        for row in active
    ]
    ok, desc = receipts.completed_gate(
        completed,
        expected_slots=("a",),
        member_receipts=active,
        expected_member_count=2,
    )
    assert not ok and "members=" in desc


def test_completed_gate_rejects_invalid_distributed_active_membership():
    duplicate_rank = [
        {"pid": 10, "rank": 0, "world_size": 2, "slots": ["a"]},
        {"pid": 11, "rank": 0, "world_size": 2, "slots": ["a"]},
    ]
    wrong_world = [
        {"pid": 10, "rank": 0, "world_size": 8, "slots": ["a"]},
        {"pid": 11, "rank": 1, "world_size": 8, "slots": ["a"]},
    ]
    for active in (duplicate_rank, wrong_world):
        completed = [
            {"slot": "a", "pid": row["pid"], "rank": row["rank"],
             "world_size": row["world_size"]}
            for row in active
        ]
        ok, desc = receipts.completed_gate(
            completed,
            expected_slots=("a",),
            member_receipts=active,
            expected_member_count=2,
        )
        assert not ok and "malformed" in desc


def test_completed_gate_rejects_identity_change_for_active_pid():
    active = _active_members(2)
    completed = _completed_members(2)
    completed[0] = {**completed[0], "rank": 1}
    ok, desc = receipts.completed_gate(
        completed,
        expected_slots=("a", "b"),
        member_receipts=active,
        expected_member_count=2,
    )
    assert not ok and "malformed" in desc


def test_unknown_active_tp_members_require_completion_rank_proof():
    active = [
        {"pid": 10, "rank": -1, "world_size": -1, "slots": ["a"]},
        {"pid": 11, "rank": -1, "world_size": -1, "slots": ["a"]},
    ]

    def completed(identities):
        return [
            {"slot": "a", "pid": 10 + index, "rank": rank, "world_size": world}
            for index, (rank, world) in enumerate(identities)
        ]

    for invalid in (
        completed(((0, 2), (0, 2))),
        completed(((0, 8), (1, 8))),
        completed(((-1, -1), (-1, -1))),
    ):
        ok, desc = receipts.completed_gate(
            invalid,
            expected_slots=("a",),
            member_receipts=active,
            expected_member_count=2,
        )
        assert not ok and "malformed" in desc

    ok, desc = receipts.completed_gate(
        completed(((0, 2), (1, 2))),
        expected_slots=("a",),
        member_receipts=active,
        expected_member_count=2,
    )
    assert ok and "2/2" in desc


def test_coverage_without_active_members_uses_consistent_world_size():
    detail = receipts.coverage_matrix(
        [{"slot": "a", "pid": 11, "rank": 1, "world_size": 2}],
        expected_slots=("a",),
        expected_member_count=2,
    )
    assert not detail["ok"]
    assert detail["basis"] == "rank"
    assert detail["missing"] == [{"slot": "a", "member": "rank:0"}]


def test_coverage_rejects_unproven_or_conflicting_members():
    unproven = receipts.coverage_matrix(
        [{"slot": "a", "pid": 10, "rank": -1, "world_size": -1}],
        expected_slots=("a",),
    )
    conflicting = receipts.coverage_matrix(
        [
            {"slot": "a", "pid": 10, "rank": 0, "world_size": 1},
            {"slot": "a", "pid": 11, "rank": 1, "world_size": 2},
        ],
        expected_slots=("a",),
    )
    assert not unproven["ok"] and unproven["basis"] == "unproven"
    assert not conflicting["ok"] and conflicting["malformed"]


@pytest.mark.parametrize(
    "bad",
    (
        {"slot": "a", "pid": True, "rank": 0, "world_size": 1},
        {"slot": "a", "pid": "10", "rank": 0, "world_size": 1},
        {"slot": "a", "pid": 10, "rank": 2, "world_size": 2},
    ),
)
def test_coverage_rejects_non_exact_or_incoherent_identity(bad):
    detail = receipts.coverage_matrix(
        [bad], expected_slots=("a",), member_receipts=_active_members(1)
    )
    assert not detail["ok"] and detail["malformed"]


def test_coverage_rejects_duplicate_and_unexpected_completion():
    active = _active_members(1)
    duplicate = [
        {"slot": "a", "pid": 10, "rank": 0, "world_size": 1},
        {"slot": "a", "pid": 10, "rank": 0, "world_size": 1},
    ]
    unexpected = [
        {"slot": "other", "pid": 10, "rank": 0, "world_size": 1}
    ]
    assert not receipts.coverage_matrix(
        duplicate, expected_slots=("a",), member_receipts=active
    )["ok"]
    assert not receipts.coverage_matrix(
        unexpected, expected_slots=("a",), member_receipts=active
    )["ok"]


def test_any_fallback_disqualifies_complete_execution():
    active = _active_members(1)
    completed = [{"slot": "a", "pid": 10, "rank": 0, "world_size": 1}]
    ok, desc = receipts.completed_gate(
        completed,
        expected_slots=("a",),
        member_receipts=active,
        expected_member_count=1,
        fallback_receipts=[{"unexpected": "still disqualifying"}],
    )
    assert not ok and "fallbacks" in desc
