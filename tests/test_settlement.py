from __future__ import annotations

from dataclasses import replace

import pytest

from optima.discovery import DiscoveryArmPlan
from optima.eval.evidence_store import EvidenceArtifactRef
from optima.eval.oci_session_protocol import SlotAuditPolicy
from optima.settlement import (
    SettlementCandidate,
    SettlementError,
    SettlementEvidence,
    SettlementEvent,
    SettlementEventType,
    SettlementQualification,
    plan_settlement,
)
from optima.stack_identity import sha256_hex
from optima.stack_manifest import (
    EvaluationStackContext,
    EvaluationStackManifest,
    ProposalContributionRef,
)
from optima.stack_plan import plan_marginal_arm
from optima.target_catalog import (
    MOE_EPILOGUE_ATOMIC_TARGET,
    MOE_EPILOGUE_MEMBERS,
    TargetCatalog,
    default_target_catalog,
)


MSA = "attention.msa_prefill_block_score"
SILU = "activation.silu_and_mul"


def _h(label: str) -> str:
    return sha256_hex(label.encode())


def _audit_policy(label: str, slots: tuple[str, ...]) -> SlotAuditPolicy:
    return SlotAuditPolicy(_h(f"audit-seed:{label}")[:32], 100_000, 32, slots, 1)


def _context(catalog: TargetCatalog) -> EvaluationStackContext:
    rows = catalog.snapshot()["targets"]
    assert isinstance(rows, list)
    return EvaluationStackContext(
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("base"),
        arena_digest=_h("arena"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        target_spec_digests={
            row["target_id"]: catalog.target_spec_digest(row["target_id"])
            for row in rows
        },
    )


def _stack(
    catalog: TargetCatalog,
    entries: dict[str, ProposalContributionRef] | None = None,
) -> EvaluationStackManifest:
    return EvaluationStackManifest(
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("base"),
        arena_digest=_h("arena"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries=entries or {},
    )


def _ref(catalog: TargetCatalog, target: str, label: str) -> ProposalContributionRef:
    return ProposalContributionRef(
        target_id=target,
        target_spec_digest=catalog.target_spec_digest(target),
        artifact_digest=_h(f"artifact:{label}"),
        selected_payload_digest=_h(f"payload:{label}"),
        attribution_digest=_h(f"attribution:{label}"),
    )


def _candidate(
    incumbent: EvaluationStackManifest,
    replacement: ProposalContributionRef,
    catalog: TargetCatalog,
    *,
    label: str,
    speedup: str = "1.05",
    block: int = 10,
    event: int = 0,
) -> SettlementCandidate:
    plan = plan_marginal_arm(
        incumbent,
        replacement,
        catalog=catalog,
        incumbent_tree_digest=_h("incumbent-tree"),
        candidate_tree_digest=_h(f"candidate-tree:{label}"),
        expected_context=_context(catalog),
    )
    members = catalog.require(replacement.target_id).members
    audit_slots = tuple(sorted(members))
    primary_audit = _audit_policy(f"primary:{label}", audit_slots)
    primary = SettlementQualification(
        lane="registered",
        arena_digest=incumbent.arena_digest,
        reservation_digest=_h(f"reservation:{label}"),
        finalized_block=block,
        event_index=event,
        event_subindex=0,
        hotkey=f"miner-{label}",
        target_id=replacement.target_id,
        members=tuple(sorted(members)),
        selected_delta_digest=plan.selected_delta_digest,
        qualification_authority_digest=_h(f"authority:{label}"),
        qualification_plan_digest=_h(f"plan-authority:{label}"),
        qualification_attempt_digest=_h(f"attempt:{label}"),
        qualification_report_digest=_h(f"report:{label}"),
        selection_commitment_digest=_h(f"selection-commitment:{label}"),
        selection_secret_commitment_digest=_h(f"selection-secret:{label}"),
        selection_evidence_digest=_h(f"selection-evidence:{label}"),
        arm_digest=plan.digest,
        incumbent_stack_digest=plan.baseline_before.stack_digest,
        incumbent_tree_digest=plan.baseline_before.tree_digest,
        candidate_stack_digest=plan.challenger.stack_digest,
        candidate_tree_digest=plan.challenger.tree_digest,
        speedup=speedup,
        incumbent_manifest=incumbent,
        candidate_manifest=plan.candidate,
        audit_control_digest=primary_audit.control.digest,
        audit_policy=primary_audit,
        audit_evidence_digest=_h(f"audit-evidence:{label}"),
    )
    reproduction_audit = _audit_policy(f"reproduction:{label}", audit_slots)
    reproduction = replace(
        primary,
        qualification_authority_digest=_h(f"reproduction-authority:{label}"),
        qualification_plan_digest=_h(f"reproduction-plan-authority:{label}"),
        qualification_attempt_digest=_h(f"reproduction-attempt:{label}"),
        qualification_report_digest=_h(f"reproduction-report:{label}"),
        selection_commitment_digest=_h(f"reproduction-selection-commitment:{label}"),
        selection_secret_commitment_digest=_h(f"reproduction-selection-secret:{label}"),
        selection_evidence_digest=_h(f"reproduction-selection-evidence:{label}"),
        audit_policy=reproduction_audit,
        audit_evidence_digest=_h(f"reproduction-audit-evidence:{label}"),
        speedup=("1.04" if speedup == "1.05" else speedup),
    )
    return SettlementCandidate.from_reproductions(primary, reproduction)


def _discovery(
    incumbent: EvaluationStackManifest, *, label: str = "d", block: int = 10
) -> SettlementCandidate:
    arm = DiscoveryArmPlan.create(
        incumbent=incumbent,
        incumbent_tree_digest=_h("incumbent-tree"),
        candidate_tree_digest=_h(f"discovery-tree:{label}"),
        proposal_digest=_h(f"proposal:{label}"),
        policy_digest=_h("discovery-policy"),
        build_profile_digest=_h("build-profile"),
        overlay_identity_digest=_h(f"overlay:{label}"),
    )
    audit_slots = ("sglang.inference.v1",)
    primary_audit = _audit_policy(f"primary:{label}", audit_slots)
    primary = SettlementQualification(
        lane="discovery",
        arena_digest=incumbent.arena_digest,
        reservation_digest=_h(f"reservation:{label}"),
        finalized_block=block,
        event_index=0,
        event_subindex=0,
        hotkey=f"miner-{label}",
        target_id="sglang.inference.v1",
        members=("sglang.inference.v1",),
        selected_delta_digest=arm.selected_delta_digest,
        qualification_authority_digest=_h(f"authority:{label}"),
        qualification_plan_digest=_h(f"plan-authority:{label}"),
        qualification_attempt_digest=_h(f"attempt:{label}"),
        qualification_report_digest=_h(f"report:{label}"),
        selection_commitment_digest=_h(f"selection-commitment:{label}"),
        selection_secret_commitment_digest=_h(f"selection-secret:{label}"),
        selection_evidence_digest=_h(f"selection-evidence:{label}"),
        arm_digest=arm.digest,
        incumbent_stack_digest=arm.baseline_before.stack_digest,
        incumbent_tree_digest=arm.baseline_before.tree_digest,
        candidate_stack_digest=arm.challenger.stack_digest,
        candidate_tree_digest=arm.challenger.tree_digest,
        speedup="1.03",
        incumbent_manifest=incumbent,
        proposal_digest=arm.proposal_digest,
        audit_control_digest=primary_audit.control.digest,
        audit_policy=primary_audit,
        audit_evidence_digest=_h(f"audit-evidence:{label}"),
    )
    reproduction_audit = _audit_policy(f"reproduction:{label}", audit_slots)
    return SettlementCandidate.from_reproductions(
        primary,
        replace(
            primary,
            qualification_authority_digest=_h(f"reproduction-authority:{label}"),
            qualification_plan_digest=_h(f"reproduction-plan-authority:{label}"),
            qualification_attempt_digest=_h(f"reproduction-attempt:{label}"),
            qualification_report_digest=_h(f"reproduction-report:{label}"),
            selection_commitment_digest=_h(
                f"reproduction-selection-commitment:{label}"
            ),
            selection_secret_commitment_digest=_h(
                f"reproduction-selection-secret:{label}"
            ),
            selection_evidence_digest=_h(f"reproduction-selection-evidence:{label}"),
            audit_policy=reproduction_audit,
            audit_evidence_digest=_h(f"reproduction-audit-evidence:{label}"),
            speedup="1.02",
        ),
    )


def test_candidate_json_round_trip_and_digest_are_canonical() -> None:
    catalog = default_target_catalog()
    candidate = _candidate(_stack(catalog), _ref(catalog, MSA, "a"), catalog, label="a")
    reopened = SettlementCandidate.from_dict(candidate.to_dict())
    assert reopened == candidate
    assert reopened.digest == candidate.digest
    assert candidate.speedup == "1.04"
    with pytest.raises(SettlementError, match="canonical decimal"):
        replace(candidate.primary, speedup="1.050")
    with pytest.raises(SettlementError, match="target/delta"):
        replace(candidate.primary, selected_delta_digest=_h("other"))


def test_single_pass_or_reused_evidence_cannot_become_settlement_candidate() -> None:
    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, MSA, "a"), catalog, label="a"
    )
    with pytest.raises(SettlementError, match="reuses primary"):
        SettlementCandidate.from_reproductions(candidate.primary, candidate.primary)
    with pytest.raises(SettlementError, match="reproduction identity"):
        SettlementCandidate.from_reproductions(
            candidate.primary,
            replace(candidate.reproduction, hotkey="different-miner"),
        )


@pytest.mark.parametrize(
    "field",
    (
        "qualification_authority_digest",
        "qualification_plan_digest",
        "qualification_attempt_digest",
        "qualification_report_digest",
        "selection_commitment_digest",
        "selection_secret_commitment_digest",
        "selection_evidence_digest",
    ),
)
def test_each_reproduction_authority_and_evidence_identity_must_be_distinct(
    field: str,
) -> None:
    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, MSA, "a"), catalog, label="a"
    )
    reproduced = replace(
        candidate.reproduction,
        **{field: getattr(candidate.primary, field)},
    )
    with pytest.raises(SettlementError, match="reuses primary"):
        SettlementCandidate.from_reproductions(candidate.primary, reproduced)


def test_conservative_speed_uses_slower_independent_reproduction() -> None:
    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, MSA, "a"), catalog,
        label="a", speedup="1.09",
    )
    slower = replace(candidate.reproduction, speedup="1.03")
    reproduced = SettlementCandidate.from_reproductions(candidate.primary, slower)
    assert reproduced.speedup == "1.03"


def test_reproduction_must_use_the_same_speed_evidence_policy() -> None:
    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, MSA, "a"), catalog, label="a"
    )
    mismatched = replace(
        candidate.reproduction,
        speed_evidence_policy_digest=_h("repeat-speed-policy"),
    )
    with pytest.raises(SettlementError, match="contribution identity"):
        SettlementCandidate.from_reproductions(candidate.primary, mismatched)


@pytest.mark.parametrize("field", ("sample_rate_ppm", "minimum_calls"))
def test_reproduction_must_use_the_same_seed_independent_audit_control(
    field: str,
) -> None:
    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, MSA, "a"), catalog, label="a"
    )
    assert candidate.reproduction.audit_policy is not None
    changed_policy = replace(
        candidate.reproduction.audit_policy,
        **{
            field: getattr(candidate.reproduction.audit_policy, field) + 1,
        },
    )
    mismatched = replace(
        candidate.reproduction,
        audit_control_digest=changed_policy.control.digest,
        audit_policy=changed_policy,
    )
    with pytest.raises(SettlementError, match="contribution identity"):
        SettlementCandidate.from_reproductions(candidate.primary, mismatched)


def test_reproduction_requires_distinct_audit_seed_and_evidence() -> None:
    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, MSA, "a"), catalog, label="a"
    )
    assert candidate.primary.audit_policy is not None
    reused_seed = replace(
        candidate.reproduction,
        audit_policy=replace(
            candidate.reproduction.audit_policy,
            validator_seed=candidate.primary.audit_policy.validator_seed,
        ),
    )
    with pytest.raises(SettlementError, match="slot-audit authority"):
        SettlementCandidate.from_reproductions(candidate.primary, reused_seed)
    reused_evidence = replace(
        candidate.reproduction,
        audit_evidence_digest=candidate.primary.audit_evidence_digest,
    )
    with pytest.raises(SettlementError, match="slot-audit authority"):
        SettlementCandidate.from_reproductions(candidate.primary, reused_evidence)


def test_new_candidate_rejects_legacy_auditless_qualification_but_reopens_history() -> None:
    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, MSA, "a"), catalog, label="a"
    )
    legacy_primary = replace(
        candidate.primary,
        audit_control_digest=SettlementQualification.__dataclass_fields__[  # type: ignore[index]
            "audit_control_digest"
        ].default,
        audit_policy=None,
        audit_evidence_digest=SettlementQualification.__dataclass_fields__[  # type: ignore[index]
            "audit_evidence_digest"
        ].default,
    )
    legacy_reproduction = replace(
        candidate.reproduction,
        audit_control_digest=legacy_primary.audit_control_digest,
        audit_policy=None,
        audit_evidence_digest=legacy_primary.audit_evidence_digest,
    )
    with pytest.raises(SettlementError, match="requires two audited"):
        SettlementCandidate.from_reproductions(
            legacy_primary, legacy_reproduction
        )
    historical = SettlementCandidate(legacy_primary, legacy_reproduction)
    assert SettlementCandidate.from_dict(historical.to_dict()) == historical


def test_settlement_evidence_binds_both_retained_attempts() -> None:
    catalog = default_target_catalog()
    candidate = _candidate(
        _stack(catalog), _ref(catalog, MSA, "a"), catalog, label="a"
    )
    primary_ref = EvidenceArtifactRef(
        "qualification.cohort-attempt",
        candidate.primary.qualification_attempt_digest,
        1,
        "application/json",
        "optima.qualification.cohort-attempt.v1",
    )
    reproduction_ref = EvidenceArtifactRef(
        "qualification.cohort-attempt",
        candidate.reproduction.qualification_attempt_digest,
        1,
        "application/json",
        "optima.qualification.cohort-attempt.v1",
    )
    evidence = SettlementEvidence.bind(
        candidate,
        primary_attempt_ref=primary_ref,
        reproduction_attempt_ref=reproduction_ref,
    )
    assert SettlementEvidence.from_dict(evidence.to_dict()) == evidence
    with pytest.raises(SettlementError, match="differ from the candidate"):
        SettlementEvidence.bind(
            candidate,
            primary_attempt_ref=reproduction_ref,
            reproduction_attempt_ref=primary_ref,
        )


def test_highest_speedup_wins_and_events_form_hash_chain() -> None:
    catalog = default_target_catalog()
    incumbent = _stack(catalog)
    early = _candidate(
        incumbent, _ref(catalog, MSA, "early"), catalog,
        label="early", speedup="1.04", block=10,
    )
    late = _candidate(
        incumbent, _ref(catalog, MSA, "late"), catalog,
        label="late", speedup="1.06", block=11,
    )
    plan = plan_settlement(
        (early, late), current_manifest=incumbent,
        current_tree_digest=_h("incumbent-tree"), initial_event_sequence=7,
    )
    assert plan.winner_candidate_digest == late.digest
    assert plan.transition is not None
    assert plan.transition.manifest == late.candidate_manifest
    assert [row.event_type for row in plan.events] == [
        SettlementEventType.HOLD,
        SettlementEventType.CROWN,
        SettlementEventType.ADOPTION,
        SettlementEventType.STACK_TRANSITION,
    ]
    assert plan.events[0].reason == "conflict_lost"
    assert [row.sequence for row in plan.events] == [7, 8, 9, 10]
    for prior, current in zip(plan.events, plan.events[1:]):
        assert current.previous_event_digest == prior.digest
    assert SettlementEvent.from_dict(plan.events[1].to_dict()) == plan.events[1]


def test_equal_speedup_uses_finalized_order_not_input_order() -> None:
    catalog = default_target_catalog()
    incumbent = _stack(catalog)
    first = _candidate(
        incumbent, _ref(catalog, MSA, "first"), catalog,
        label="first", speedup="1.05", block=10, event=2,
    )
    second = _candidate(
        incumbent, _ref(catalog, MSA, "second"), catalog,
        label="second", speedup="1.05", block=11, event=0,
    )
    plan = plan_settlement(
        (second, first), current_manifest=incumbent,
        current_tree_digest=_h("incumbent-tree"),
    )
    assert plan.winner_candidate_digest == first.digest


def test_stale_candidate_holds_without_stack_change() -> None:
    catalog = default_target_catalog()
    old = _stack(catalog)
    candidate = _candidate(old, _ref(catalog, MSA, "old"), catalog, label="old")
    current_ref = _ref(catalog, SILU, "current")
    current = _stack(catalog, {SILU: current_ref})
    plan = plan_settlement(
        (candidate,), current_manifest=current, current_tree_digest=_h("other-tree")
    )
    assert plan.transition is None
    assert plan.before == plan.after
    assert [row.event_type for row in plan.events] == [SettlementEventType.HOLD]
    assert plan.events[0].reason == "stale_incumbent"


def test_discovery_pass_only_creates_bounty_and_never_changes_stack() -> None:
    catalog = default_target_catalog()
    incumbent = _stack(catalog)
    candidate = _discovery(incumbent)
    plan = plan_settlement(
        (candidate,), current_manifest=incumbent,
        current_tree_digest=_h("incumbent-tree"),
    )
    assert plan.transition is None
    assert plan.winner_candidate_digest == ""
    assert [row.event_type for row in plan.events] == [
        SettlementEventType.DISCOVERY_BOUNTY
    ]


def test_nonoverlapping_loser_is_held_for_requalification() -> None:
    catalog = default_target_catalog()
    incumbent = _stack(catalog)
    msa = _candidate(
        incumbent, _ref(catalog, MSA, "msa"), catalog,
        label="msa", speedup="1.07",
    )
    silu = _candidate(
        incumbent, _ref(catalog, SILU, "silu"), catalog,
        label="silu", speedup="1.06",
    )
    plan = plan_settlement(
        (silu, msa), current_manifest=incumbent,
        current_tree_digest=_h("incumbent-tree"),
    )
    hold = next(row for row in plan.events if row.event_type is SettlementEventType.HOLD)
    assert hold.candidate_digest == silu.digest
    assert hold.reason == "incumbent_advanced"


def test_replacement_retires_prior_and_atomic_transition_neutralizes_displaced() -> None:
    catalog = default_target_catalog()
    prior = _ref(catalog, MSA, "prior")
    incumbent = _stack(catalog, {MSA: prior})
    replacement = _candidate(
        incumbent, _ref(catalog, MSA, "next"), catalog, label="next"
    )
    replaced = plan_settlement(
        (replacement,), current_manifest=incumbent,
        current_tree_digest=_h("incumbent-tree"),
    )
    assert SettlementEventType.RETIREMENT in {
        row.event_type for row in replaced.events
    }

    singleton_entries = {
        member: _ref(catalog, member, f"prior:{member}")
        for member in MOE_EPILOGUE_MEMBERS
    }
    atomic_incumbent = _stack(catalog, singleton_entries)
    atomic = _candidate(
        atomic_incumbent,
        _ref(catalog, MOE_EPILOGUE_ATOMIC_TARGET, "atomic"),
        catalog,
        label="atomic",
    )
    transitioned = plan_settlement(
        (atomic,), current_manifest=atomic_incumbent,
        current_tree_digest=_h("incumbent-tree"),
    )
    neutralized = {
        row.target_id
        for row in transitioned.events
        if row.event_type is SettlementEventType.NEUTRALIZATION
    }
    assert neutralized == set(MOE_EPILOGUE_MEMBERS)


def test_duplicate_reservation_is_rejected() -> None:
    catalog = default_target_catalog()
    incumbent = _stack(catalog)
    candidate = _candidate(
        incumbent, _ref(catalog, MSA, "one"), catalog, label="one"
    )
    with pytest.raises(SettlementError, match="duplicates"):
        plan_settlement(
            (candidate, candidate), current_manifest=incumbent,
            current_tree_digest=_h("incumbent-tree"),
        )
