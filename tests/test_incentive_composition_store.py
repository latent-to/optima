from __future__ import annotations

from types import SimpleNamespace

import pytest

from optima.chain.debt_publication import (
    PUBLICATION_KIND_COMPOSED,
    build_debt_weight_publication_binding,
)
from optima.chain.finite_debt_store import reward_family_id
from optima.chain.incentive_activation import selected_model_campaign_id
from optima.chain.incentive_composition_store import (
    IncentiveCompositionStoreError,
    SELECTED_CORE_SELECTION_REPORT_DIGEST,
    SELECTED_SELECTION_REPORT_DIGEST,
    SelectedIncentiveActivationApproval,
)
from optima.chain.intake import IntakeError
from optima.chain.weights import WeightPublicationRecord
from optima.finite_debt import (
    CampaignBudgetShare,
    IMPROVEMENT_GROSS,
    PPM,
    FiniteDebtPolicyManifest,
    RewardFamilyCampaign,
    pay_claim_balance,
    project_debt_epoch,
)
from optima.incentive_composition import (
    DISCOVERY_BOUNTY_ONLY,
    DISCOVERY_REGISTERED_PROMOTION,
    IncentiveCompositionPolicyManifest,
    pay_discovery_balance,
    review_discovery_disposition,
)
from optima.settlement import SettlementCandidate
from tests.test_chain_intake import (
    _h,
    _qualified_discovery_candidate,
    _qualified_settlement_candidate,
    _store,
)
from tests.test_finite_debt_store import (
    _commit,
    _confirmed_debt_publication,
    _family,
)


_APPROVED_ARENA_DIGEST = _h("selected incentive activation arena")
_APPROVED_CATALOG_DIGEST = _h("selected incentive activation catalog")
_APPROVED_EVALUATION_STACK_DIGEST = _h(
    "selected incentive activation evaluation stack"
)
_APPROVED_MEMBERSHIP_DIGEST = _h("selected incentive activation membership")
_APPROVED_AUDIT_CONTROL_MANIFEST_DIGEST = _h(
    "selected incentive activation audit controls"
)
_APPROVED_AUDIT_CANARY_RECEIPT_DIGEST = _h(
    "selected incentive activation audit canary"
)
_APPROVED_AUDIT_RISK_ACCEPTANCE_DIGEST = _h(
    "selected incentive activation audit residual-risk acceptance"
)


def _selected_core(family_id: str) -> FiniteDebtPolicyManifest:
    campaign_id = selected_model_campaign_id(
        arena_digest=_APPROVED_ARENA_DIGEST,
        catalog_digest=_APPROVED_CATALOG_DIGEST,
        reward_family_ids=(family_id,),
    )
    return FiniteDebtPolicyManifest(
        campaign_budget_shares=(CampaignBudgetShare(campaign_id, PPM),),
        reward_family_campaigns=(RewardFamilyCampaign(family_id, campaign_id),),
        selection_report_digest=SELECTED_CORE_SELECTION_REPORT_DIGEST,
        reserve_hotkey="reserve",
        reserve_ppm=100_000,
        epoch_blocks=7_200,
        beta_ppm=100_000,
        tau_blocks=648_000,
        lifetime_blocks=648_000,
        k_ppm=PPM,
        improvement_basis=IMPROVEMENT_GROSS,
        clock_reset_threshold_log_units_ppm=1,
    )


def _selected_composition(
    core: FiniteDebtPolicyManifest,
) -> IncentiveCompositionPolicyManifest:
    return IncentiveCompositionPolicyManifest(
        innovation_policy_digest=core.digest,
        selection_report_digest=SELECTED_SELECTION_REPORT_DIGEST,
        reserve_ppm=100_000,
        epoch_blocks=7_200,
        discovery_cap_units=50_000,
        per_award_principal_cap_epochs=1,
        discovery_lifetime_blocks=648_000,
    )


def _approval(store, core, composition, *, block: int = 10):
    block_hash = "0x" + f"{block:064x}"
    return SelectedIncentiveActivationApproval(
        store.scope.digest,
        core.digest,
        composition.digest,
        core.campaign_budget_shares[0].campaign_id,
        _APPROVED_ARENA_DIGEST,
        _APPROVED_EVALUATION_STACK_DIGEST,
        _APPROVED_CATALOG_DIGEST,
        _APPROVED_MEMBERSHIP_DIGEST,
        _APPROVED_AUDIT_CONTROL_MANIFEST_DIGEST,
        _APPROVED_AUDIT_CANARY_RECEIPT_DIGEST,
        _APPROVED_AUDIT_RISK_ACCEPTANCE_DIGEST,
        core.family_ids,
        core.reserve_hotkey,
        block,
        block_hash,
    )


def _activate_core(store, core, *, block: int = 10):
    block_hash = "0x" + f"{block:064x}"
    store.reserve_finalized((), finalized_block=block, finalized_block_hash=block_hash)
    composition = _selected_composition(core)
    approval = _approval(store, core, composition, block=block)
    activation = store.activate_selected_incentives(
        core,
        composition,
        approval,
        expected_approval_digest=approval.digest,
    )
    assert activation.approval == approval
    return composition, activation


def _activate_selected(store, candidate):
    family_id = candidate if isinstance(candidate, str) else _family(candidate)
    core = _selected_core(family_id)
    composition, activation = _activate_core(store, core)
    return core, composition, activation


def _default_family() -> str:
    from optima.target_catalog import default_target_catalog

    catalog = default_target_catalog()
    target = "activation.silu_and_mul"
    return reward_family_id(
        _h("arena"), target, catalog.target_spec_digest(target)
    )


def _composition_publication_binding(store, activation):
    boundary = (
        activation.activation_block + activation.policy.epoch_blocks
    )
    projection = store.project_incentive_composition_epoch(
        effective_block=boundary,
        eligible_hotkeys=("reserve",),
    )
    return build_debt_weight_publication_binding(
        projection,
        publication_kind=PUBLICATION_KIND_COMPOSED,
        activation_digest=activation.digest,
        chain_scope_digest=store.scope.digest,
        netuid=store.scope.netuid,
        validator_hotkey="validator",
        boundary_metagraph=SimpleNamespace(
            block=boundary,
            block_hash="0x" + f"{boundary:064x}",
            hotkeys=["reserve", "validator"],
            uids=[0, 1],
        ),
        epoch_index=1,
    )


def _review(
    policy,
    win,
    *,
    marker: str,
    block: int,
    decision: str,
    hotkey: str | None = None,
):
    return review_discovery_disposition(
        policy,
        win_digest=win.digest,
        proposal_digest=win.proposal_digest,
        retained_evidence_digest=win.retained_evidence_digest,
        review_digest=_h(f"review:{marker}"),
        hotkey=win.hotkey if hotkey is None else hotkey,
        win_block=win.settlement_block,
        authority_block=block,
        decision=decision,
        requested_principal_epochs=7 if decision == DISCOVERY_BOUNTY_ONLY else 0,
        promoted_target_digest=(
            _h(f"target:{marker}")
            if decision == DISCOVERY_REGISTERED_PROMOTION
            else ""
        ),
    )


def _retain_discovery_win(store, *, marker: str):
    from optima.settlement import plan_settlement

    core = _selected_core(_h(f"unused lifecycle family:{marker}"))
    policy, _activation = _activate_core(store, core)
    candidate = _qualified_discovery_candidate(
        store,
        index=1,
        proposal_digest=_h(f"lifecycle:{marker}"),
        hotkey="lifecycle-discoverer",
    )
    lease = store.lease_settlement_cohort(current_block=11)
    assert lease is not None and lease.candidates == (candidate,)
    plan = plan_settlement(
        lease.candidates,
        current_manifest=lease.stack.manifest,
        current_tree_digest=lease.stack.tree_digest,
        initial_event_sequence=lease.initial_event_sequence,
        previous_event_digest=lease.previous_event_digest,
    )
    evidence = tuple(
        store.reopen_settlement_evidence(row) for row in lease.candidates
    )
    block11 = "0x" + f"{11:064x}"
    store.reserve_finalized(
        (), finalized_block=11, finalized_block_hash=block11
    )
    store.commit_settlement(
        lease,
        plan,
        evidence,
        current_block=11,
        current_block_hash=block11,
    )
    win = store.review_pending_discovery_wins()[0]
    return candidate, policy, win


def test_schema4_to5_is_empty_no_retro_and_immutable(tmp_path) -> None:
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        for table in (
            "incentive_composed_allocations",
            "incentive_composed_epochs",
            "incentive_discovery_balances",
            "incentive_discovery_claims",
            "incentive_discovery_dispositions",
            "incentive_discovery_wins",
            "incentive_composition_activations",
        ):
            store._db.execute(f"DROP TABLE {table}")
        store._db.execute("UPDATE metadata SET value='4' WHERE key='schema'")

    with _store(tmp_path) as reopened:
        assert reopened._db.execute(
            "SELECT value FROM metadata WHERE key='schema'"
        ).fetchone()["value"] == "5"
        assert reopened.reviewed_discovery_dispositions() == ()
        assert reopened.discovery_debt_claim_states() == ()
        assert reopened.incentive_composition_reward_epochs() == ()
        assert reopened.active_incentive_composition(at_block=10) is None
        triggers = {
            row["name"]
            for row in reopened._db.execute(
                "SELECT name FROM sqlite_schema WHERE type='trigger' "
                "AND name LIKE 'incentive_%_reject_%'"
            )
        }
        assert len(triggers) == 14


def test_activation_is_exact_and_legacy_discovery_fails_closed(tmp_path) -> None:
    with _store(tmp_path) as store:
        core = _selected_core(_default_family())
        block_hash = "0x" + f"{10:064x}"
        store.reserve_finalized((), finalized_block=10, finalized_block_hash=block_hash)
        with pytest.raises(IntakeError, match="standalone finite-debt activation"):
            store.activate_finite_debt_policy(
                core,
                activation_block=10,
                activation_block_hash=block_hash,
            )
        wrong = IncentiveCompositionPolicyManifest(
            innovation_policy_digest=core.digest,
            selection_report_digest=_h("wrong selection report"),
            reserve_ppm=100_000,
            epoch_blocks=7_200,
            discovery_cap_units=50_000,
            per_award_principal_cap_epochs=1,
            discovery_lifetime_blocks=648_000,
        )
        with pytest.raises(IntakeError, match="standalone composition activation"):
            store.activate_incentive_composition(
                wrong,
                activation_block=10,
                activation_block_hash=block_hash,
            )
        wrong_approval = _approval(store, core, wrong)
        with pytest.raises(IntakeError, match="exact D-013"):
            store.activate_selected_incentives(
                core,
                wrong,
                wrong_approval,
                expected_approval_digest=wrong_approval.digest,
            )
        assert store._finite_debt.policy_activations() == ()
        selected = _selected_composition(core)
        selected_approval = _approval(store, core, selected)
        with pytest.raises(IntakeError, match="pinned digest"):
            store.activate_selected_incentives(
                core,
                selected,
                selected_approval,
                expected_approval_digest=_h("different operator approval"),
            )
        assert store._finite_debt.policy_activations() == ()
        store._db.execute(
            "INSERT INTO discovery_bounty_claims(claim_digest,proposal_digest,"
            "claim_json,status,event_id) VALUES(?,?,?,'active',?)",
            (_h("legacy claim"), _h("legacy proposal"), "{}", _h("legacy event")),
        )
        with pytest.raises(
            IntakeError, match="discovery reward claim is corrupt|legacy discovery"
        ):
            selected = _selected_composition(core)
            approval = _approval(store, core, selected)
            store.activate_selected_incentives(
                core,
                selected,
                approval,
                expected_approval_digest=approval.digest,
            )
        store._db.execute(
            "UPDATE discovery_bounty_claims SET status='forged_terminal'"
        )
        with pytest.raises(IntakeError, match="legacy discovery"):
            selected = _selected_composition(core)
            approval = _approval(store, core, selected)
            store.activate_selected_incentives(
                core,
                selected,
                approval,
                expected_approval_digest=approval.digest,
            )

    stale_core_root = tmp_path / "stale-core-selection"
    with _store(stale_core_root) as store:
        selected = _selected_core(_default_family())
        stale = FiniteDebtPolicyManifest(
            **{
                **{
                    field: getattr(selected, field)
                    for field in selected.__dataclass_fields__
                },
                "selection_report_digest": _h("stale D-012 selection report"),
            }
        )
        block_hash = "0x" + f"{10:064x}"
        store.reserve_finalized((), finalized_block=10, finalized_block_hash=block_hash)
        composition = _selected_composition(stale)
        approval = _approval(store, stale, composition)
        with pytest.raises(IntakeError, match="exact D-013"):
            store.activate_selected_incentives(
                stale,
                composition,
                approval,
                expected_approval_digest=approval.digest,
            )
        assert store._finite_debt.policy_activations() == ()


def test_active_composition_raises_schema_floor_and_rejects_runtime_rollback(
    tmp_path,
) -> None:
    with _store(tmp_path) as store:
        _activate_selected(store, _default_family())
        assert store._db.execute(
            "SELECT value FROM metadata WHERE key='schema'"
        ).fetchone()["value"] == "6"
        store._db.execute(
            "UPDATE metadata SET value='5' WHERE key='schema' AND value='6'"
        )

    with pytest.raises(IntakeError, match="schema-6 rollback fencing"):
        with _store(tmp_path):
            pass


def test_schema6_without_selected_activation_fails_closed(tmp_path) -> None:
    with _store(tmp_path) as store:
        store._db.execute(
            "UPDATE metadata SET value='6' WHERE key='schema' AND value='5'"
        )

    with pytest.raises(IntakeError, match="exactly one selected activation"):
        with _store(tmp_path):
            pass


def test_selected_composition_rejects_a_second_model_campaign(tmp_path) -> None:
    with _store(tmp_path) as store:
        family_a = _default_family()
        family_b = _h("second model family")
        campaign_a = selected_model_campaign_id(
            arena_digest=_APPROVED_ARENA_DIGEST,
            catalog_digest=_APPROVED_CATALOG_DIGEST,
            reward_family_ids=tuple(sorted((family_a, family_b))),
        )
        campaign_b = _h("second model campaign")
        core = FiniteDebtPolicyManifest(
            campaign_budget_shares=(
                CampaignBudgetShare(campaign_a, 500_000),
                CampaignBudgetShare(campaign_b, 500_000),
            ),
            reward_family_campaigns=(
                RewardFamilyCampaign(family_a, campaign_a),
                RewardFamilyCampaign(family_b, campaign_b),
            ),
            selection_report_digest=SELECTED_CORE_SELECTION_REPORT_DIGEST,
            reserve_hotkey="reserve",
            reserve_ppm=100_000,
            epoch_blocks=7_200,
            beta_ppm=100_000,
            tau_blocks=648_000,
            lifetime_blocks=648_000,
            k_ppm=PPM,
            improvement_basis=IMPROVEMENT_GROSS,
            clock_reset_threshold_log_units_ppm=1,
        )
        block_hash = "0x" + f"{10:064x}"
        store.reserve_finalized((), finalized_block=10, finalized_block_hash=block_hash)
        composition = _selected_composition(core)
        approval = SelectedIncentiveActivationApproval(
            store.scope.digest,
            core.digest,
            composition.digest,
            campaign_a,
            _APPROVED_ARENA_DIGEST,
            _APPROVED_EVALUATION_STACK_DIGEST,
            _APPROVED_CATALOG_DIGEST,
            _APPROVED_MEMBERSHIP_DIGEST,
            _APPROVED_AUDIT_CONTROL_MANIFEST_DIGEST,
            _APPROVED_AUDIT_CANARY_RECEIPT_DIGEST,
            _APPROVED_AUDIT_RISK_ACCEPTANCE_DIGEST,
            core.family_ids,
            core.reserve_hotkey,
            10,
            block_hash,
        )
        with pytest.raises(IntakeError, match="exact D-013"):
            store.activate_selected_incentives(
                core,
                composition,
                approval,
                expected_approval_digest=approval.digest,
            )
        assert store._finite_debt.policy_activations() == ()


def test_legacy_standing_title_survives_composition_without_retro_debt(tmp_path) -> None:
    from optima.economics import (
        EmissionsPolicyManifest,
        GlobalRewardProjectionContext,
        MetagraphMember,
    )

    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        _commit(store, candidate, with_hash=False)
        standing_before = store.active_reward_claims()[0]
        crown_before = store.reopen_active_crown(
            candidate.arena_digest, candidate.target_id
        )
        assert len(standing_before) == 1

        core = _selected_core(_family(candidate))
        block12 = "0x" + f"{12:064x}"
        _composition, activation = _activate_core(store, core, block=12)
        assert activation.approval.reward_family_ids == (_family(candidate),)
        clocks = store.finite_debt_family_clocks(policy_digest=core.digest)
        assert len(clocks) == 1 and clocks[0].source == "seed"
        assert store.active_reward_claims()[0] == standing_before
        assert store.reopen_active_crown(
            candidate.arena_digest, candidate.target_id
        ) == crown_before
        assert store.finite_debt_claim_states() == ()

        context = GlobalRewardProjectionContext(
            store.scope.digest,
            "validator",
            12,
            block12,
            (MetagraphMember(0, "validator"),),
        )
        with pytest.raises(IntakeError, match="legacy V1 weight projection"):
            store.build_weight_projection(
                policy=EmissionsPolicyManifest(100, 20, 100_000),
                context=context,
                catalogs={},
                netuid=store.scope.netuid,
            )


def test_composed_disposition_projection_close_and_restart(tmp_path) -> None:
    with _store(tmp_path) as store:
        core, _composition, activation = _activate_selected(store, _default_family())
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        _commit(store, candidate, current_block=12)

        core_before = store.finite_debt_claim_states()[0]
        core_only_projection = project_debt_epoch(
            core,
            effective_block=7_210,
            states=(core_before,),
        )
        projection = store.project_incentive_composition_epoch(
            effective_block=7_210,
            eligible_hotkeys=("miner", "reserve"),
        )
        assert store.finite_debt_claim_states()[0] == core_before
        assert store.discovery_debt_claim_states() == ()
        assert projection.discovery_payout_units == 0
        assert projection.innovation_payout_units == 900_000
        assert projection.reserve_units == 100_000
        assert sum(row.units for row in projection.weights) == PPM
        with pytest.raises(IntakeError, match="core-only projection"):
            store.project_finite_debt_epoch(
                effective_block=7_210,
                eligible_hotkeys=("miner", "reserve"),
            )

        boundary_hash = "0x" + f"{7_210:064x}"
        store.reserve_finalized(
            (), finalized_block=7_210, finalized_block_hash=boundary_hash
        )
        confirmation = _confirmed_debt_publication(
            store,
            projection,
            activation,
            publication_kind=PUBLICATION_KIND_COMPOSED,
        )
        epoch = store.close_confirmed_composed_epoch(
            projection,
            confirmation=confirmation,
            eligible_hotkeys=("miner", "reserve"),
        )
        core_after = store.finite_debt_claim_states()[0]
        assert (
            core_after.balance.paid_units - core_before.balance.paid_units
            == 900_000
        )
        assert epoch.activation_digest == activation.digest
        assert store.close_confirmed_composed_epoch(
            projection,
            confirmation=confirmation,
            eligible_hotkeys=("miner", "reserve"),
        ) == epoch
        with pytest.raises(IntakeError, match="core-only close"):
            store.close_confirmed_debt_epoch(
                core_only_projection,
                confirmation=confirmation,
                eligible_hotkeys=("miner", "reserve"),
            )
        events = store.finite_debt_reward_events()
        assert [row["event_type"] for row in events] == [
            "policy_activated",
            "composition_policy_activated",
            "claim_issued",
            "composed_epoch_paid",
        ]

    with _store(tmp_path) as reopened:
        assert reopened.incentive_composition_reward_epochs() == (epoch,)
        assert reopened.finite_debt_claim_states()[0] == core_after
        assert reopened.discovery_debt_claim_states() == ()
        assert reopened.project_incentive_composition_epoch(
            effective_block=7_210,
            eligible_hotkeys=("miner", "reserve"),
        ) == projection


@pytest.mark.parametrize(
    ("status", "record_fields"),
    (
        ("intent", {"submit_block": 7_210, "retry_after_block": 7_220}),
        ("pending", {"submit_block": 7_210, "retry_after_block": 7_220}),
        ("held", {}),
        ("released", {"reason": "operator released retry hold"}),
        (
            "confirmed",
            {"confirmed_block": 7_210, "confirmed_last_update": 7_210},
        ),
    ),
)
def test_every_unclosed_v2_publication_status_fences_economic_mutation(
    tmp_path, status: str, record_fields: dict[str, object]
) -> None:
    with _store(tmp_path) as store:
        _core, _composition, activation = _activate_selected(
            store, _default_family()
        )
        binding = _composition_publication_binding(store, activation)
        boundary = binding.economic_projection.effective_block
        store.reserve_finalized(
            (),
            finalized_block=boundary,
            finalized_block_hash=binding.effective_block_hash,
        )
        journal = store.debt_weight_publication_journal(binding)
        journal.compare_and_swap(
            None,
            WeightPublicationRecord(
                binding.weight_projection.digest,
                status,
                reason=record_fields.get("reason", f"retained {status}"),
                submit_block=record_fields.get("submit_block", 0),
                retry_after_block=record_fields.get("retry_after_block", 0),
                confirmed_block=record_fields.get("confirmed_block", 0),
                confirmed_last_update=record_fields.get(
                    "confirmed_last_update", 0
                ),
            ),
        )
        assert store.unclosed_debt_publication_bindings() == (binding,)
        with pytest.raises(IntakeError, match="unclosed V2 debt publication"):
            store.reconcile_incentive_composition_lifecycle(
                current_block=10,
                current_block_hash="0x" + f"{10:064x}",
                eligible_hotkeys=("reserve",),
            )


def test_dry_run_binding_without_journal_row_does_not_fence_settlement(
    tmp_path,
) -> None:
    with _store(tmp_path) as store:
        _core, _composition, activation = _activate_selected(
            store, _default_family()
        )
        binding = _composition_publication_binding(store, activation)
        journal = store.debt_weight_publication_journal(binding)
        assert journal.load() is None
        assert store.unclosed_debt_publication_bindings() == ()

        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        assert store.has_pending_settlement()
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None and lease.candidates == (candidate,)


def test_first_publication_cas_reprojects_after_pre_intent_state_change(
    tmp_path,
) -> None:
    with _store(tmp_path, expiry_blocks=10_000) as store:
        _core, _composition, activation = _activate_selected(
            store, _default_family()
        )
        stale_binding = _composition_publication_binding(store, activation)
        journal = store.debt_weight_publication_journal(stale_binding)
        assert journal.load() is None

        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        _commit(store, candidate, current_block=11)
        boundary = stale_binding.economic_projection.effective_block
        store.reserve_finalized(
            (),
            finalized_block=boundary,
            finalized_block_hash=stale_binding.effective_block_hash,
        )

        with pytest.raises(
            IntakeError, match="economic state changed before V2 publication intent"
        ):
            journal.compare_and_swap(
                None,
                WeightPublicationRecord(
                    stale_binding.weight_projection.digest,
                    "intent",
                    submit_block=boundary,
                    retry_after_block=boundary + 10,
                    reason="must reproject under BEGIN IMMEDIATE",
                ),
            )
        assert journal.load() is None
        assert store.unclosed_debt_publication_bindings() == ()


def test_publication_fence_survives_restart_and_holds_pending_settlement(
    tmp_path,
) -> None:
    with _store(tmp_path, expiry_blocks=10_000) as store:
        _core, _composition, activation = _activate_selected(
            store, _default_family()
        )
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        binding = _composition_publication_binding(store, activation)
        boundary = binding.economic_projection.effective_block
        store.reserve_finalized(
            (),
            finalized_block=boundary,
            finalized_block_hash=binding.effective_block_hash,
        )
        journal = store.debt_weight_publication_journal(binding)
        journal.compare_and_swap(
            None,
            WeightPublicationRecord(
                binding.weight_projection.digest,
                "intent",
                submit_block=7_210,
                retry_after_block=7_220,
                reason="before signer call",
            ),
        )
        assert not store.has_pending_settlement()
        assert store.lease_settlement_cohort(current_block=boundary) is None
        assert store._db.execute(
            "SELECT status FROM settlement_candidates WHERE candidate_digest=?",
            (candidate.digest,),
        ).fetchone()["status"] == "pending"

    with _store(tmp_path, expiry_blocks=10_000) as reopened:
        assert reopened.unclosed_debt_publication_bindings() == (binding,)
        assert not reopened.has_pending_settlement()
        assert reopened.lease_settlement_cohort(current_block=boundary) is None
        assert reopened._db.execute(
            "SELECT status FROM settlement_candidates WHERE candidate_digest=?",
            (candidate.digest,),
        ).fetchone()["status"] == "pending"


def test_publication_intent_after_lease_blocks_commit_without_partial_crown(
    tmp_path,
) -> None:
    from optima.settlement import plan_settlement

    with _store(tmp_path, expiry_blocks=10_000) as store:
        _core, _composition, activation = _activate_selected(
            store, _default_family()
        )
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        boundary = (
            activation.activation_block + activation.policy.epoch_blocks
        )
        lease = store.lease_settlement_cohort(current_block=boundary - 1)
        assert lease is not None and lease.candidates == (candidate,)
        plan = plan_settlement(
            lease.candidates,
            current_manifest=lease.stack.manifest,
            current_tree_digest=lease.stack.tree_digest,
            initial_event_sequence=lease.initial_event_sequence,
            previous_event_digest=lease.previous_event_digest,
        )
        evidence = tuple(
            store.reopen_settlement_evidence(row) for row in lease.candidates
        )
        boundary_hash = "0x" + f"{boundary:064x}"
        store.reserve_finalized(
            (), finalized_block=boundary, finalized_block_hash=boundary_hash
        )
        binding = _composition_publication_binding(store, activation)
        journal = store.debt_weight_publication_journal(binding)
        journal.compare_and_swap(
            None,
            WeightPublicationRecord(
                binding.weight_projection.digest,
                "intent",
                submit_block=7_210,
                retry_after_block=7_220,
                reason="raced a retained lease",
            ),
        )

        with pytest.raises(IntakeError, match="unclosed V2 debt publication"):
            store.commit_settlement(
                lease,
                plan,
                evidence,
                current_block=boundary,
                current_block_hash=boundary_hash,
            )
        assert store.finite_debt_claim_states() == ()
        assert store.active_reward_claims() == ((), ())
        assert all(
            row["event_type"] != "claim_issued"
            for row in store.finite_debt_reward_events()
        )
        assert store._db.execute(
            "SELECT status FROM settlement_candidates WHERE candidate_digest=?",
            (candidate.digest,),
        ).fetchone()["status"] == "leased"


def test_publication_fence_clears_only_after_exact_epoch_close(tmp_path) -> None:
    with _store(tmp_path) as store:
        _core, _composition, activation = _activate_selected(
            store, _default_family()
        )
        binding = _composition_publication_binding(store, activation)
        boundary = binding.economic_projection.effective_block
        boundary_hash = "0x" + f"{boundary:064x}"
        store.reserve_finalized(
            (), finalized_block=boundary, finalized_block_hash=boundary_hash
        )
        confirmation = _confirmed_debt_publication(
            store,
            binding.economic_projection,
            activation,
            publication_kind=PUBLICATION_KIND_COMPOSED,
        )
        assert store.unclosed_debt_publication_bindings() == (binding,)

        with pytest.raises(IntakeError):
            store.close_confirmed_composed_epoch(
                binding.economic_projection,
                confirmation=confirmation,
                eligible_hotkeys=("validator",),
            )
        assert store.unclosed_debt_publication_bindings() == (binding,)

        store.close_confirmed_composed_epoch(
            binding.economic_projection,
            confirmation=confirmation,
            eligible_hotkeys=("reserve",),
        )
        assert store.unclosed_debt_publication_bindings() == ()
        store.reconcile_incentive_composition_lifecycle(
            current_block=boundary,
            current_block_hash=boundary_hash,
            eligible_hotkeys=("reserve",),
        )

    with _store(tmp_path) as reopened:
        assert reopened.unclosed_debt_publication_bindings() == ()


def test_missed_boundaries_freeze_crowns_until_gapless_catch_up(tmp_path) -> None:
    from optima.settlement import plan_settlement

    with _store(tmp_path, expiry_blocks=30_000) as store:
        _core, _composition, activation = _activate_selected(
            store, _default_family()
        )
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        first_boundary = (
            activation.activation_block + activation.policy.epoch_blocks
        )
        second_boundary = first_boundary + activation.policy.epoch_blocks
        second_hash = "0x" + f"{second_boundary:064x}"
        store.reserve_finalized(
            (),
            finalized_block=second_boundary,
            finalized_block_hash=second_hash,
        )

        assert store.due_debt_publication_boundary() == first_boundary
        assert not store.has_pending_settlement()
        assert store.lease_settlement_cohort(
            current_block=second_boundary
        ) is None
        assert store.finite_debt_claim_states() == ()

        for boundary in (first_boundary, second_boundary):
            projection = store.project_incentive_composition_epoch(
                effective_block=boundary,
                eligible_hotkeys=("reserve",),
            )
            confirmation = _confirmed_debt_publication(
                store,
                projection,
                activation,
                publication_kind=PUBLICATION_KIND_COMPOSED,
                confirmed_block=boundary,
                marker=f"gapless catch-up {boundary}",
            )
            store.close_confirmed_composed_epoch(
                projection,
                confirmation=confirmation,
                eligible_hotkeys=("reserve",),
            )

        assert store.due_debt_publication_boundary() is None
        assert store.has_pending_settlement()
        lease = store.lease_settlement_cohort(current_block=second_boundary)
        assert lease is not None and lease.candidates == (candidate,)
        plan = plan_settlement(
            lease.candidates,
            current_manifest=lease.stack.manifest,
            current_tree_digest=lease.stack.tree_digest,
            initial_event_sequence=lease.initial_event_sequence,
            previous_event_digest=lease.previous_event_digest,
        )
        evidence = tuple(
            store.reopen_settlement_evidence(row) for row in lease.candidates
        )
        store.commit_settlement(
            lease,
            plan,
            evidence,
            current_block=second_boundary,
            current_block_hash=second_hash,
        )
        claim = store.finite_debt_claim_states()[0]
        assert claim.claim.settlement_block == second_boundary
        assert all(
            claim.digest not in epoch.projection.innovation_input_state_digests
            for epoch in store.incentive_composition_reward_epochs()
        )


def test_composed_catch_up_cannot_compress_live_emission_epochs(tmp_path) -> None:
    with _store(tmp_path) as store:
        _core, _composition, activation = _activate_selected(
            store, _default_family()
        )
        first = store.project_incentive_composition_epoch(
            effective_block=7_210,
            eligible_hotkeys=("reserve",),
        )
        store.reserve_finalized(
            (),
            finalized_block=7_300,
            finalized_block_hash="0x" + f"{7300:064x}",
        )
        first_confirmation = _confirmed_debt_publication(
            store,
            first,
            activation,
            publication_kind=PUBLICATION_KIND_COMPOSED,
            confirmed_block=7_300,
            marker="late first boundary",
        )
        store.close_confirmed_composed_epoch(
            first,
            confirmation=first_confirmation,
            eligible_hotkeys=("reserve",),
        )

        second = store.project_incentive_composition_epoch(
            effective_block=14_410,
            eligible_hotkeys=("reserve",),
        )
        store.reserve_finalized(
            (),
            finalized_block=14_410,
            finalized_block_hash="0x" + f"{14410:064x}",
        )
        compressed = _confirmed_debt_publication(
            store,
            second,
            activation,
            publication_kind=PUBLICATION_KIND_COMPOSED,
            confirmed_block=14_410,
            marker="compressed second boundary",
        )
        with pytest.raises(IntakeError, match="compress live emission epochs"):
            store.close_confirmed_composed_epoch(
                second,
                confirmation=compressed,
                eligible_hotkeys=("reserve",),
            )

        store.reserve_finalized(
            (),
            finalized_block=14_500,
            finalized_block_hash="0x" + f"{14500:064x}",
        )
        rate_limited = _confirmed_debt_publication(
            store,
            second,
            activation,
            publication_kind=PUBLICATION_KIND_COMPOSED,
            confirmed_block=14_500,
            marker="rate-limited second boundary",
        )
        epoch = store.close_confirmed_composed_epoch(
            second,
            confirmation=rate_limited,
            eligible_hotkeys=("reserve",),
        )
        assert epoch.effective_block == 14_410


@pytest.mark.parametrize("reward_class", ("core", "discovery"))
def test_composed_epoch_reopen_rejects_extra_revision_reusing_payout_event(
    tmp_path, reward_class: str,
) -> None:
    with _store(tmp_path) as store:
        _core, composition, activation = _activate_selected(
            store, _default_family()
        )
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        discoveries = (
            _qualified_discovery_candidate(
                store,
                index=0,
                proposal_digest=_h("extra-revision-discovery-a"),
                hotkey="discoverer-a",
            ),
            _qualified_discovery_candidate(
                store,
                index=1,
                proposal_digest=_h("extra-revision-discovery-b"),
                hotkey="discoverer-b",
            ),
        )
        from optima.settlement import plan_settlement

        # Keep the registered candidate from advancing the incumbent between
        # the two discovery-only settlements.  This is test fixture state, not
        # an economic transition; restore it before its own leased settlement.
        store._db.execute(
            "UPDATE settlement_candidates SET status='held',reason='test_fixture' "
            "WHERE candidate_digest=?",
            (candidate.digest,),
        )
        for block, expected in ((11, discoveries[0]), (12, discoveries[1])):
            lease = store.lease_settlement_cohort(current_block=block)
            assert lease is not None and lease.candidates == (expected,)
            plan = plan_settlement(
                lease.candidates,
                current_manifest=lease.stack.manifest,
                current_tree_digest=lease.stack.tree_digest,
                initial_event_sequence=lease.initial_event_sequence,
                previous_event_digest=lease.previous_event_digest,
            )
            evidence = tuple(
                store.reopen_settlement_evidence(row) for row in lease.candidates
            )
            block_hash = "0x" + f"{block:064x}"
            store.reserve_finalized(
                (), finalized_block=block, finalized_block_hash=block_hash
            )
            store.commit_settlement(
                lease,
                plan,
                evidence,
                current_block=block,
                current_block_hash=block_hash,
            )
        store._db.execute(
            "UPDATE settlement_candidates SET status='pending',reason='' "
            "WHERE candidate_digest=? AND status='held'",
            (candidate.digest,),
        )
        _commit(store, candidate, current_block=13)

        wins = store.review_pending_discovery_wins()
        assert len(wins) == 2
        for index, win in enumerate(wins, start=14):
            block_hash = "0x" + f"{index:064x}"
            store.reserve_finalized(
                (), finalized_block=index, finalized_block_hash=block_hash
            )
            store.record_reviewed_discovery_disposition(
                _review(
                    composition,
                    win,
                    marker=f"extra-revision-{index}",
                    block=index,
                    decision=DISCOVERY_BOUNTY_ONLY,
                ),
                authority_block_hash=block_hash,
            )

        eligible = ("discoverer-a", "discoverer-b", "miner", "reserve")
        projection = store.project_incentive_composition_epoch(
            effective_block=7_210,
            eligible_hotkeys=eligible,
        )
        assert projection.discovery_payout_units == 50_000
        assert projection.innovation_payout_units == 850_000
        boundary_hash = "0x" + f"{7_210:064x}"
        store.reserve_finalized(
            (), finalized_block=7_210, finalized_block_hash=boundary_hash
        )
        epoch = store.close_confirmed_composed_epoch(
            projection,
            confirmation=_confirmed_debt_publication(
                store,
                projection,
                activation,
                publication_kind=PUBLICATION_KIND_COMPOSED,
            ),
            eligible_hotkeys=eligible,
        )
        if reward_class == "core":
            after = store.finite_debt_claim_states()[0]
            forged = pay_claim_balance(
                after.claim,
                after.balance,
                1,
                at_block=7_210,
            )
            revision = store._db.execute(
                "SELECT MAX(revision) AS value FROM finite_debt_claim_balances "
                "WHERE claim_digest=?",
                (after.claim.digest,),
            ).fetchone()["value"] + 1
            with store._transaction():
                store._finite_debt._insert_balance(
                    forged,
                    revision=revision,
                    reward_event_digest=epoch.payout_event_digest,
                )
            with pytest.raises(IntakeError, match="not exactly authorized"):
                store.finite_debt_claim_states()
        else:
            after = store.discovery_debt_claim_states()[0]
            forged = pay_discovery_balance(
                after.claim,
                after.balance,
                1,
                at_block=7_210,
            )
            revision = store._db.execute(
                "SELECT MAX(revision) AS value FROM incentive_discovery_balances "
                "WHERE claim_digest=?",
                (after.claim.digest,),
            ).fetchone()["value"] + 1
            with store._transaction():
                store._incentive_composition._insert_discovery_balance(
                    forged,
                    revision=revision,
                    reward_event_digest=epoch.payout_event_digest,
                )
            with pytest.raises(IntakeError, match="not exactly authorized"):
                store.discovery_debt_claim_states()
        with pytest.raises(IntakeError, match="balance revision set differs"):
            store.incentive_composition_reward_epochs()


def test_active_composition_retains_review_pending_wins_and_binds_dispositions(
    tmp_path,
) -> None:
    with _store(tmp_path) as store:
        core = _selected_core(_h("unused selected family"))
        _composition, _activation = _activate_core(store, core)
        first = _qualified_discovery_candidate(
            store,
            index=1,
            proposal_digest=_h("post-composition discovery one"),
            hotkey="discoverer-one",
        )
        second = _qualified_discovery_candidate(
            store,
            index=2,
            proposal_digest=_h("post-composition discovery two"),
            hotkey="discoverer-two",
        )
        from optima.settlement import plan_settlement

        for block, expected in ((11, first), (12, second)):
            lease = store.lease_settlement_cohort(current_block=block)
            assert lease is not None and lease.candidates == (expected,)
            plan = plan_settlement(
                lease.candidates,
                current_manifest=lease.stack.manifest,
                current_tree_digest=lease.stack.tree_digest,
                initial_event_sequence=lease.initial_event_sequence,
                previous_event_digest=lease.previous_event_digest,
            )
            evidence = tuple(
                store.reopen_settlement_evidence(row) for row in lease.candidates
            )
            block_hash = "0x" + f"{block:064x}"
            store.reserve_finalized(
                (), finalized_block=block, finalized_block_hash=block_hash
            )
            store.commit_settlement(
                lease,
                plan,
                evidence,
                current_block=block,
                current_block_hash=block_hash,
            )

        wins = store.review_pending_discovery_wins()
        assert tuple(win.candidate_digest for win in wins) == (
            first.digest,
            second.digest,
        )
        assert store.active_reward_claims()[1] == ()
        assert store.reviewed_discovery_dispositions() == ()
        assert store.discovery_debt_claim_states() == ()

        block13 = "0x" + f"{13:064x}"
        store.reserve_finalized((), finalized_block=13, finalized_block_hash=block13)
        bounty = _review(
            _selected_composition(core),
            wins[0],
            marker="bounty",
            block=13,
            decision=DISCOVERY_BOUNTY_ONLY,
        )
        bounty_record = store.record_reviewed_discovery_disposition(
            bounty, authority_block_hash=block13
        )
        assert bounty_record.claim_digest
        assert bounty_record.disposition.decision == "bounty_only"
        assert bounty_record.disposition.review_digest == _h("review:bounty")
        claim = store.discovery_debt_claim_states()[0].claim
        assert claim.principal_units == 50_000
        assert claim.requested_principal_epochs == 7
        assert claim.capped_principal_epochs == 1
        assert claim.awarded_block == wins[0].settlement_block
        assert claim.expires_block == (
            wins[0].settlement_block + 648_000
        )

        block14 = "0x" + f"{14:064x}"
        store.reserve_finalized((), finalized_block=14, finalized_block_hash=block14)
        promotion = _review(
            _selected_composition(core),
            wins[1],
            marker="promotion",
            block=14,
            decision=DISCOVERY_REGISTERED_PROMOTION,
        )
        with pytest.raises(
            IntakeError, match="DiscoveryWinRecord/DiscoveryPromotion"
        ):
            store.record_reviewed_discovery_disposition(
                promotion, authority_block_hash=block14
            )
        assert len(store.reviewed_discovery_dispositions()) == 1
        assert len(store.discovery_debt_claim_states()) == 1

        forged = type(bounty)(
            policy_digest=bounty.policy_digest,
            win_digest=_h("forged varied win"),
            proposal_digest=_h("forged varied proposal"),
            retained_evidence_digest=wins[0].retained_evidence_digest,
            review_digest=_h("forged varied review"),
            hotkey=wins[0].hotkey,
            win_block=wins[0].settlement_block,
            authority_block=14,
            decision=DISCOVERY_BOUNTY_ONLY,
            requested_principal_epochs=1,
            promoted_target_digest="",
        )
        with pytest.raises(IntakeError, match="no retained discovery win"):
            store.record_reviewed_discovery_disposition(
                forged, authority_block_hash=block14
            )

    with _store(tmp_path) as reopened:
        assert reopened.review_pending_discovery_wins() == (wins[1],)
        assert len(reopened.reviewed_discovery_dispositions()) == 1
        assert len(reopened.discovery_debt_claim_states()) == 1


def test_discovery_bounty_cannot_refresh_or_outlive_retained_win(tmp_path) -> None:
    from optima.settlement import plan_settlement

    with _store(tmp_path) as store:
        core = _selected_core(_h("unused bounded family"))
        policy, activation = _activate_core(store, core)
        candidate = _qualified_discovery_candidate(
            store,
            index=1,
            proposal_digest=_h("bounded discovery win"),
            hotkey="bounded-discoverer",
        )
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None and lease.candidates == (candidate,)
        plan = plan_settlement(
            lease.candidates,
            current_manifest=lease.stack.manifest,
            current_tree_digest=lease.stack.tree_digest,
            initial_event_sequence=lease.initial_event_sequence,
            previous_event_digest=lease.previous_event_digest,
        )
        evidence = tuple(
            store.reopen_settlement_evidence(row) for row in lease.candidates
        )
        block11 = "0x" + f"{11:064x}"
        store.reserve_finalized((), finalized_block=11, finalized_block_hash=block11)
        store.commit_settlement(
            lease,
            plan,
            evidence,
            current_block=11,
            current_block_hash=block11,
        )
        win = store.review_pending_discovery_wins()[0]
        expiry = win.settlement_block + policy.discovery_lifetime_blocks
        expiry_hash = "0x" + f"{expiry:064x}"
        store.reserve_finalized(
            (), finalized_block=expiry, finalized_block_hash=expiry_hash
        )
        # Advancing finalized intake across 90 unpaid boundaries freezes all
        # reward-state mutation.  Catch them up gaplessly before exercising the
        # review-expiry authority at this later block.
        while (boundary := store.due_debt_publication_boundary()) is not None:
            projection = store.project_incentive_composition_epoch(
                effective_block=boundary,
                eligible_hotkeys=("reserve",),
            )
            store.close_confirmed_composed_epoch(
                projection,
                confirmation=_confirmed_debt_publication(
                    store,
                    projection,
                    activation,
                    publication_kind=PUBLICATION_KIND_COMPOSED,
                    confirmed_block=boundary,
                    marker=f"review expiry catch-up {boundary}",
                ),
                eligible_hotkeys=("reserve",),
            )

        refreshed = review_discovery_disposition(
            policy,
            win_digest=win.digest,
            proposal_digest=win.proposal_digest,
            retained_evidence_digest=win.retained_evidence_digest,
            review_digest=_h("forged refreshed win block"),
            hotkey=win.hotkey,
            win_block=expiry,
            authority_block=expiry,
            decision=DISCOVERY_BOUNTY_ONLY,
            requested_principal_epochs=1,
        )
        with pytest.raises(IntakeError, match="differs from retained candidate/evidence"):
            store.record_reviewed_discovery_disposition(
                refreshed, authority_block_hash=expiry_hash
            )

        expired = _review(
            policy,
            win,
            marker="expired bounded win",
            block=expiry,
            decision=DISCOVERY_BOUNTY_ONLY,
        )
        with pytest.raises(IntakeError, match="at or after.*expiry"):
            store.record_reviewed_discovery_disposition(
                expired, authority_block_hash=expiry_hash
            )
        assert store.reviewed_discovery_dispositions() == ()
        assert store.discovery_debt_claim_states() == ()
        assert store.expire_review_pending_discovery_wins(
            current_block=expiry,
            current_block_hash=expiry_hash,
        ) == (win,)
        assert store.expire_review_pending_discovery_wins(
            current_block=expiry,
            current_block_hash=expiry_hash,
        ) == ()
        assert store.review_pending_discovery_wins() == ()
        with pytest.raises(IntakeError, match="at or after.*expiry"):
            store.record_reviewed_discovery_disposition(
                expired, authority_block_hash=expiry_hash
            )
        assert store._db.execute(
            "SELECT status FROM settlement_candidates WHERE candidate_digest=?",
            (candidate.digest,),
        ).fetchone()["status"] == "review_expired"
        assert [
            row["event_type"] for row in store.finite_debt_reward_events()
        ][-1] == "discovery_review_expired"

    with _store(tmp_path) as reopened:
        assert reopened.review_pending_discovery_wins() == ()
        assert reopened._db.execute(
            "SELECT status FROM settlement_candidates WHERE candidate_digest=?",
            (candidate.digest,),
        ).fetchone()["status"] == "review_expired"


@pytest.mark.parametrize(
    "corruption",
    ("pending_as_bounty", "bounty_as_pending", "expired_without_event"),
)
def test_retained_win_reopen_validates_lifecycle_before_status_filter(
    tmp_path, corruption: str,
) -> None:
    with _store(tmp_path) as store:
        candidate, policy, win = _retain_discovery_win(
            store,
            marker=corruption,
        )
        if corruption == "pending_as_bounty":
            store._db.execute(
                "UPDATE settlement_candidates SET status='reviewed_bounty',"
                "reason='reviewed_bounty' WHERE candidate_digest=?",
                (candidate.digest,),
            )
        elif corruption == "bounty_as_pending":
            block12 = "0x" + f"{12:064x}"
            store.reserve_finalized(
                (), finalized_block=12, finalized_block_hash=block12
            )
            store.record_reviewed_discovery_disposition(
                _review(
                    policy,
                    win,
                    marker="lifecycle-cardinality",
                    block=12,
                    decision=DISCOVERY_BOUNTY_ONLY,
                ),
                authority_block_hash=block12,
            )
            store._db.execute(
                "UPDATE settlement_candidates SET status='review_pending',"
                "reason='review_pending' WHERE candidate_digest=?",
                (candidate.digest,),
            )
        else:
            store._db.execute(
                "UPDATE settlement_candidates SET status='review_expired',reason=? "
                "WHERE candidate_digest=?",
                (
                    f"review_expired:{_h('missing lifecycle expiry event')}",
                    candidate.digest,
                ),
            )

    with _store(tmp_path) as reopened:
        expected = (
            "reviewed discovery bounty lifecycle cardinality"
            if corruption == "pending_as_bounty"
            else "review-pending discovery lifecycle cardinality"
            if corruption == "bounty_as_pending"
            else "expired discovery review event authority differs"
        )
        with pytest.raises(IntakeError, match=expected):
            reopened.review_pending_discovery_wins()
        with pytest.raises(IncentiveCompositionStoreError, match=expected):
            reopened._incentive_composition._win_by_digest(win.digest)
        if corruption == "bounty_as_pending":
            with pytest.raises(IntakeError, match=expected):
                reopened.reviewed_discovery_dispositions()


def test_review_pending_win_reopens_exact_typed_settlement_event(tmp_path) -> None:
    from optima.settlement import plan_settlement

    with _store(tmp_path) as store:
        core = _selected_core(_h("unused event-bound family"))
        _composition, _activation = _activate_core(store, core)
        candidate = _qualified_discovery_candidate(
            store,
            index=1,
            proposal_digest=_h("event-bound discovery win"),
            hotkey="event-bound-discoverer",
        )
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None and lease.candidates == (candidate,)
        plan = plan_settlement(
            lease.candidates,
            current_manifest=lease.stack.manifest,
            current_tree_digest=lease.stack.tree_digest,
            initial_event_sequence=lease.initial_event_sequence,
            previous_event_digest=lease.previous_event_digest,
        )
        evidence = tuple(
            store.reopen_settlement_evidence(row) for row in lease.candidates
        )
        block11 = "0x" + f"{11:064x}"
        store.reserve_finalized((), finalized_block=11, finalized_block_hash=block11)
        store.commit_settlement(
            lease,
            plan,
            evidence,
            current_block=11,
            current_block_hash=block11,
        )
        win = store.review_pending_discovery_wins()[0]
        store._db.execute(
            "UPDATE settlement_events SET event_json='{}' WHERE event_digest=?",
            (win.settlement_event_digest,),
        )
        with pytest.raises(IntakeError, match="settlement event is corrupt"):
            store.review_pending_discovery_wins()


def test_core_policy_upgrade_and_legacy_v1_projection_publication_are_fenced(
    tmp_path,
) -> None:
    from optima.chain.intake import SQLiteWeightPublicationJournal
    from optima.chain.weights import WeightProjection, WeightPublicationRecord
    from optima.economics import (
        EmissionsPolicyManifest,
        GlobalRewardProjectionContext,
        MetagraphMember,
    )
    from optima.target_catalog import default_target_catalog

    with _store(tmp_path / "active") as store:
        _core, _composition, _activation = _activate_selected(
            store, _default_family()
        )
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        block20 = "0x" + f"{20:064x}"
        store.reserve_finalized((), finalized_block=20, finalized_block_hash=block20)
        with pytest.raises(IntakeError, match="standalone finite-debt activation"):
            store.activate_finite_debt_policy(
                _selected_core(_h("different selected family")),
                activation_block=20,
                activation_block_hash=block20,
            )

        context = GlobalRewardProjectionContext(
            store.scope.digest,
            "validator",
            20,
            block20,
            (MetagraphMember(0, "validator"),),
        )
        with pytest.raises(IntakeError, match="legacy V1 weight projection"):
            store.build_weight_projection(
                policy=EmissionsPolicyManifest(100, 20, 100_000),
                context=context,
                catalogs={candidate.arena_digest: default_target_catalog()},
                netuid=store.scope.netuid,
            )

        projection = WeightProjection(
            _h("scope"),
            307,
            "validator",
            _h("legacy policy"),
            _h("legacy settlement"),
            _h("legacy evaluation"),
            _h("legacy metagraph"),
            (_h("legacy arena"),),
            0,
            20,
            0,
            (),
            (("reserve", PPM),),
        )
        with pytest.raises(IntakeError, match="legacy V1 weight publication"):
            SQLiteWeightPublicationJournal(store, projection)

    with _store(tmp_path / "retained-object") as store:
        core = _selected_core(_default_family())
        projection = WeightProjection(
            _h("object scope"),
            307,
            "validator",
            _h("object policy"),
            _h("object settlement"),
            _h("object evaluation"),
            _h("object metagraph"),
            (_h("object arena"),),
            0,
            10,
            0,
            (),
            (("reserve", PPM),),
        )
        retained_journal = SQLiteWeightPublicationJournal(store, projection)
        _composition, _activation = _activate_core(store, core)
        intent = WeightPublicationRecord(
            projection.digest,
            "intent",
            submit_block=10,
            retry_after_block=20,
            reason="must be fenced after cutover",
        )
        with pytest.raises(IntakeError, match="legacy V1 weight publication"):
            retained_journal.compare_and_swap(None, intent)
        assert store._db.execute(
            "SELECT COUNT(*) AS n FROM weight_publications"
        ).fetchone()["n"] == 0

    with _store(tmp_path / "existing-journal") as store:
        core = _selected_core(_default_family())
        block10 = "0x" + f"{10:064x}"
        store.reserve_finalized((), finalized_block=10, finalized_block_hash=block10)
        projection = WeightProjection(
            _h("existing scope"),
            307,
            "validator",
            _h("existing policy"),
            _h("existing settlement"),
            _h("existing evaluation"),
            _h("existing metagraph"),
            (_h("existing arena"),),
            0,
            10,
            0,
            (),
            (("reserve", PPM),),
        )
        journal = SQLiteWeightPublicationJournal(store, projection)
        journal.compare_and_swap(
            None,
            WeightPublicationRecord(
                projection.digest,
                "pending",
                submit_block=10,
                retry_after_block=20,
                reason="unresolved legacy publication",
            ),
        )
        composition = _selected_composition(core)
        approval = _approval(store, core, composition)
        with pytest.raises(IntakeError, match="explicit cutover"):
            store.activate_selected_incentives(
                core,
                composition,
                approval,
                expected_approval_digest=approval.digest,
            )


def test_atomic_cutover_requires_quiescence_and_exact_replay_is_idempotent(
    tmp_path,
) -> None:
    with _store(tmp_path, expiry_blocks=10_000) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        core = _selected_core(_family(candidate))
        composition = _selected_composition(core)
        approval = _approval(store, core, composition)

        with pytest.raises(IntakeError, match="quiescent pre-activation intake"):
            store.activate_selected_incentives(
                core,
                composition,
                approval,
                expected_approval_digest=approval.digest,
            )
        assert store._finite_debt.policy_activations() == ()
        assert store.active_incentive_composition(at_block=10) is None

        _commit(store, candidate, current_block=11)
        activation_block = 12
        block_hash = "0x" + f"{activation_block:064x}"
        store.reserve_finalized(
            (),
            finalized_block=activation_block,
            finalized_block_hash=block_hash,
        )
        approval = SelectedIncentiveActivationApproval(
            store.scope.digest,
            core.digest,
            composition.digest,
            core.campaign_budget_shares[0].campaign_id,
            _APPROVED_ARENA_DIGEST,
            _APPROVED_EVALUATION_STACK_DIGEST,
            _APPROVED_CATALOG_DIGEST,
            _APPROVED_MEMBERSHIP_DIGEST,
            _APPROVED_AUDIT_CONTROL_MANIFEST_DIGEST,
            _APPROVED_AUDIT_CANARY_RECEIPT_DIGEST,
            _APPROVED_AUDIT_RISK_ACCEPTANCE_DIGEST,
            core.family_ids,
            core.reserve_hotkey,
            activation_block,
            block_hash,
        )
        first = store.activate_selected_incentives(
            core,
            composition,
            approval,
            expected_approval_digest=approval.digest,
        )
        replay = store.activate_selected_incentives(
            core,
            composition,
            approval,
            expected_approval_digest=approval.digest,
        )
        assert replay == first
        assert len(store._finite_debt.policy_activations()) == 1
        assert len(store._incentive_composition.policy_activations()) == 1


def test_active_composition_disables_burn_weight_projection(tmp_path) -> None:
    from optima.economics import (
        EmissionsPolicyManifest,
        GlobalRewardProjectionContext,
        MetagraphMember,
    )

    with _store(tmp_path) as store:
        core = _selected_core(_default_family())
        _activate_core(store, core)
        context = GlobalRewardProjectionContext(
            store.scope.digest,
            "validator",
            10,
            "0x" + f"{10:064x}",
            (MetagraphMember(0, "owner-burn"), MetagraphMember(1, "validator")),
        )
        with pytest.raises(IntakeError, match="legacy V1 weight projection"):
            store.build_burn_weight_projection(
                policy=EmissionsPolicyManifest(100, 20, 100_000),
                context=context,
                netuid=store.scope.netuid,
                burn_hotkey="owner-burn",
            )
