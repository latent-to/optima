from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from optima.chain.incentive_activation import (
    IncentiveActivationError,
    execute_selected_incentive_activation,
    load_selected_incentive_activation_bundle,
    selected_model_campaign_id,
)
from optima.chain.finite_debt_store import reward_family_id
from optima.chain.intake import EvaluationStackState, FinalizedIntakeStore, IntakeScope
from optima.chain.incentive_composition_store import (
    SELECTED_CORE_SELECTION_REPORT_DIGEST,
    SELECTED_SELECTION_REPORT_DIGEST,
    SelectedIncentiveActivationApproval,
)
from optima.finite_debt import (
    IMPROVEMENT_GROSS,
    PPM,
    CampaignBudgetShare,
    FiniteDebtPolicyManifest,
    RewardFamilyCampaign,
)
from optima.incentive_composition import IncentiveCompositionPolicyManifest
from optima.stack_identity import canonical_digest, canonical_json_bytes, sha256_hex
from optima.stack_manifest import EvaluationStackManifest
from optima.target_catalog import default_target_catalog


def _h(value: str) -> str:
    return sha256_hex(value.encode("utf-8"))


def _manifests(
    *,
    chain_scope_digest: str | None = None,
    family_ids: tuple[str, ...] | None = None,
    arena_digest: str | None = None,
    evaluation_stack_digest: str | None = None,
    catalog_digest: str | None = None,
    membership_digest: str | None = None,
):
    families = (
        (_h("minimax-m3 launch family"),)
        if family_ids is None
        else tuple(family_ids)
    )
    arena = _h("minimax-m3 launch arena") if arena_digest is None else arena_digest
    catalog = (
        _h("minimax-m3 launch catalog")
        if catalog_digest is None
        else catalog_digest
    )
    evaluation_stack = (
        _h("minimax-m3 launch evaluation stack")
        if evaluation_stack_digest is None
        else evaluation_stack_digest
    )
    membership = (
        _h("minimax-m3 launch membership")
        if membership_digest is None
        else membership_digest
    )
    campaign = selected_model_campaign_id(
        arena_digest=arena,
        catalog_digest=catalog,
        reward_family_ids=families,
    )
    core = FiniteDebtPolicyManifest(
        campaign_budget_shares=(CampaignBudgetShare(campaign, PPM),),
        reward_family_campaigns=tuple(
            RewardFamilyCampaign(family, campaign) for family in families
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
    composition = IncentiveCompositionPolicyManifest(
        innovation_policy_digest=core.digest,
        selection_report_digest=SELECTED_SELECTION_REPORT_DIGEST,
        reserve_ppm=100_000,
        epoch_blocks=7_200,
        discovery_cap_units=50_000,
        per_award_principal_cap_epochs=1,
        discovery_lifetime_blocks=648_000,
    )
    approval = SelectedIncentiveActivationApproval(
        _h("chain scope") if chain_scope_digest is None else chain_scope_digest,
        core.digest,
        composition.digest,
        campaign,
        arena,
        evaluation_stack,
        catalog,
        membership,
        _h("minimax-m3 launch audit control manifest"),
        _h("minimax-m3 launch audit canary receipt"),
        _h("minimax-m3 launch audit residual-risk acceptance"),
        families,
        core.reserve_hotkey,
        100,
        "0x" + f"{100:064x}",
    )
    return core, composition, approval


def _write(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _paths(tmp_path: Path):
    core, composition, approval = _manifests()
    core_path = tmp_path / "core.json"
    composition_path = tmp_path / "composition.json"
    approval_path = tmp_path / "approval.json"
    _write(core_path, core.to_dict())
    _write(composition_path, composition.to_dict())
    _write(approval_path, approval.to_dict())
    return core, composition, approval, core_path, composition_path, approval_path


def test_activation_bundle_reopens_exact_reviewed_bytes(tmp_path: Path) -> None:
    core, composition, approval, core_path, composition_path, approval_path = _paths(
        tmp_path
    )
    bundle = load_selected_incentive_activation_bundle(
        core_policy_path=core_path,
        composition_policy_path=composition_path,
        approval_path=approval_path,
        expected_approval_digest=approval.digest,
    )
    assert bundle.core_policy == core
    assert bundle.composition_policy == composition
    assert bundle.approval == approval
    assert len(bundle.digest) == 64


def test_activation_bundle_rejects_unpinned_or_cross_bound_manifests(
    tmp_path: Path,
) -> None:
    core, _composition, approval, core_path, composition_path, approval_path = _paths(
        tmp_path
    )
    with pytest.raises(IncentiveActivationError, match="pinned approval"):
        load_selected_incentive_activation_bundle(
            core_policy_path=core_path,
            composition_policy_path=composition_path,
            approval_path=approval_path,
            expected_approval_digest=_h("wrong approval"),
        )

    other_campaign = _h("other campaign")
    other = FiniteDebtPolicyManifest.from_dict(
        {
            **core.to_dict(),
            "campaign_budget_shares": [
                {"campaign_id": other_campaign, "share_ppm": PPM}
            ],
            "reward_family_campaigns": [
                {
                    "campaign_id": other_campaign,
                    "family_id": core.family_ids[0],
                }
            ],
        }
    )
    _write(core_path, other.to_dict())
    with pytest.raises(IncentiveActivationError, match="pinned approval"):
        load_selected_incentive_activation_bundle(
            core_policy_path=core_path,
            composition_policy_path=composition_path,
            approval_path=approval_path,
            expected_approval_digest=approval.digest,
        )


def test_activation_bundle_rejects_noncanonical_duplicate_or_symlink_inputs(
    tmp_path: Path,
) -> None:
    _core, _composition, approval, core_path, composition_path, approval_path = _paths(
        tmp_path
    )
    core_path.write_text('{"z":1, "a":2}\n', encoding="utf-8")
    with pytest.raises(IncentiveActivationError, match="canonically encoded"):
        load_selected_incentive_activation_bundle(
            core_policy_path=core_path,
            composition_policy_path=composition_path,
            approval_path=approval_path,
            expected_approval_digest=approval.digest,
        )

    core_path.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(IncentiveActivationError, match="duplicate key"):
        load_selected_incentive_activation_bundle(
            core_policy_path=core_path,
            composition_policy_path=composition_path,
            approval_path=approval_path,
            expected_approval_digest=approval.digest,
        )

    target = tmp_path / "target.json"
    _write(target, _manifests()[0].to_dict())
    core_path.unlink()
    os.symlink(target, core_path)
    with pytest.raises(IncentiveActivationError, match="cannot open core policy"):
        load_selected_incentive_activation_bundle(
            core_policy_path=core_path,
            composition_policy_path=composition_path,
            approval_path=approval_path,
            expected_approval_digest=approval.digest,
        )


class _Subtensor:
    def __init__(self, *, wrong_historical_hash: bool = False) -> None:
        self.wrong_historical_hash = wrong_historical_hash

    def get_block_hash(self, block: int) -> str:
        if block == 0:
            return "0x" + "0" * 64
        if block == 100 and not self.wrong_historical_hash:
            return "0x" + f"{100:064x}"
        return "0x" + f"{block + 1:064x}"


def _membership_digest(
    scope: IntakeScope,
    *,
    hotkeys: tuple[str, ...] = ("validator", "reserve"),
    uids: tuple[int, ...] = (0, 1),
) -> str:
    members = [
        {"hotkey": hotkey, "uid": uid}
        for uid, hotkey in sorted(
            zip(uids, hotkeys, strict=True), key=lambda item: (item[0], item[1])
        )
    ]
    return canonical_digest(
        "optima.economics.metagraph-membership",
        {
            "block": 100,
            "block_hash": "0x" + f"{100:064x}",
            "chain_scope_digest": scope.digest,
            "members": members,
        },
    )


def _evaluation_state(
    manifest: EvaluationStackManifest, *, tree_digest: str
) -> EvaluationStackState:
    transition_event_id = canonical_digest(
        "optima.chain.evaluation-stack-genesis",
        {
            "arena_digest": manifest.arena_digest,
            "stack_digest": manifest.digest,
            "tree_digest": tree_digest,
        },
    )
    return EvaluationStackState(
        manifest.arena_digest,
        0,
        manifest,
        tree_digest,
        transition_event_id,
    )


def _execution_paths(tmp_path: Path, *, retained_fault: str | None = None):
    scope = IntakeScope("0x" + "0" * 64, 307)
    catalog = default_target_catalog()
    arena = _h("minimax-m3 production arena")
    tree_digest = _h("minimax tree")
    manifest = EvaluationStackManifest(
        runtime_digest=_h("minimax runtime"),
        base_engine_digest=_h("minimax base engine"),
        arena_digest=arena,
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries={},
    )
    expected_state = _evaluation_state(manifest, tree_digest=tree_digest)
    targets = catalog.snapshot()["targets"]
    assert isinstance(targets, list)
    families = tuple(
        sorted(
            reward_family_id(
                arena,
                row["target_id"],
                catalog.target_spec_digest(row["target_id"]),
            )
            for row in targets
        )
    )
    core, composition, approval = _manifests(
        chain_scope_digest=scope.digest,
        family_ids=families,
        arena_digest=arena,
        evaluation_stack_digest=expected_state.digest,
        catalog_digest=catalog.digest,
        membership_digest=_membership_digest(scope),
    )
    core_path = tmp_path / "core.json"
    composition_path = tmp_path / "composition.json"
    approval_path = tmp_path / "approval.json"
    _write(core_path, core.to_dict())
    _write(composition_path, composition.to_dict())
    _write(approval_path, approval.to_dict())
    database = tmp_path / "private" / "intake.sqlite3"
    with FinalizedIntakeStore(database, scope=scope) as store:
        retained_manifest = manifest
        retained_tree_digest = tree_digest
        if retained_fault == "arena":
            retained_manifest = EvaluationStackManifest(
                runtime_digest=manifest.runtime_digest,
                base_engine_digest=manifest.base_engine_digest,
                arena_digest=_h("different retained arena"),
                catalog_snapshot=catalog.snapshot(),
                catalog_digest=catalog.digest,
                entries={},
            )
        elif retained_fault == "stack":
            retained_manifest = EvaluationStackManifest(
                runtime_digest=_h("different retained runtime"),
                base_engine_digest=manifest.base_engine_digest,
                arena_digest=arena,
                catalog_snapshot=catalog.snapshot(),
                catalog_digest=catalog.digest,
                entries={},
            )
            retained_tree_digest = _h("different retained tree")
        elif retained_fault == "catalog":
            snapshot = catalog.snapshot()
            snapshot["policy_version"] = "target-catalog.v1-different"
            retained_catalog_digest = canonical_digest(
                "optima.target-catalog", snapshot
            )
            retained_manifest = EvaluationStackManifest(
                runtime_digest=manifest.runtime_digest,
                base_engine_digest=manifest.base_engine_digest,
                arena_digest=arena,
                catalog_snapshot=snapshot,
                catalog_digest=retained_catalog_digest,
                entries={},
            )
        elif retained_fault is not None:
            raise AssertionError(f"unsupported retained fault {retained_fault!r}")
        store.initialize_evaluation_stack(
            retained_manifest, tree_digest=retained_tree_digest
        )
        store.reserve_finalized(
            (),
            finalized_block=approval.activation_block,
            finalized_block_hash=approval.activation_block_hash,
        )
    return scope, core, composition, approval, database, arena, (
        core_path,
        composition_path,
        approval_path,
    )


def test_execute_activation_is_wallet_free_exact_cursor_and_idempotent(
    tmp_path: Path,
) -> None:
    scope, core, composition, approval, database, arena, paths = _execution_paths(
        tmp_path
    )

    def metagraph(_subtensor, netuid, *, block=None):
        height = 100 if block is None else block
        return type(
            "Metagraph",
            (),
            {
                "netuid": netuid,
                "block": height,
                "block_hash": "0x" + f"{height:064x}",
                "uids": [0, 1],
                "hotkeys": ["validator", "reserve"],
            },
        )()

    def execute():
        return execute_selected_incentive_activation(
            network="mock",
            netuid=scope.netuid,
            intake_db=database,
            core_policy_path=paths[0],
            composition_policy_path=paths[1],
            approval_path=paths[2],
            expected_approval_digest=approval.digest,
            connect=lambda network: _Subtensor() if network == "mock" else None,
            read_finalized_head=lambda _subtensor: (
                101,
                "0x" + f"{101:064x}",
            ),
            fetch_metagraph=metagraph,
        )

    first = execute()
    second = execute()
    assert second == first
    assert first.chain_scope_digest == scope.digest
    assert first.approval_digest == approval.digest
    assert first.campaign_id == core.campaign_budget_shares[0].campaign_id
    assert first.reward_family_ids == core.family_ids
    assert first.arena_digest == approval.arena_digest == arena
    assert first.evaluation_stack_digest == approval.evaluation_stack_digest
    assert first.catalog_digest == approval.catalog_digest
    assert first.membership_digest == approval.membership_digest
    assert (
        first.audit_control_manifest_digest
        == approval.audit_control_manifest_digest
    )
    assert first.audit_canary_receipt_digest == approval.audit_canary_receipt_digest
    assert (
        first.audit_residual_risk_acceptance_digest
        == approval.audit_residual_risk_acceptance_digest
    )
    assert len(first.digest) == 64
    with FinalizedIntakeStore(database, scope=scope) as store:
        active = store.active_incentive_composition(at_block=100)
        assert active is not None
        assert active.digest == first.activation_digest
        assert active.policy == composition
        assert active.approval.arena_digest == first.arena_digest
        assert (
            active.approval.evaluation_stack_digest
            == first.evaluation_stack_digest
        )
        assert active.approval.catalog_digest == first.catalog_digest
        assert active.approval.membership_digest == first.membership_digest
        assert (
            active.approval.audit_control_manifest_digest
            == first.audit_control_manifest_digest
        )
        assert (
            active.approval.audit_canary_receipt_digest
            == first.audit_canary_receipt_digest
        )
        assert (
            active.approval.audit_residual_risk_acceptance_digest
            == first.audit_residual_risk_acceptance_digest
        )


@pytest.mark.parametrize(
    ("cursor_block", "wrong_historical_hash", "message"),
    [
        (101, False, "intake cursor differs"),
        (100, True, "not on the finalized chain"),
    ],
)
def test_execute_activation_rejects_cursor_or_chain_ancestry_mismatch(
    tmp_path: Path,
    cursor_block: int,
    wrong_historical_hash: bool,
    message: str,
) -> None:
    scope, _core, _composition, approval, database, _arena, paths = _execution_paths(
        tmp_path
    )
    if cursor_block != approval.activation_block:
        with FinalizedIntakeStore(database, scope=scope) as store:
            store.reserve_finalized(
                (),
                finalized_block=cursor_block,
                finalized_block_hash="0x" + f"{cursor_block:064x}",
            )
    with pytest.raises(IncentiveActivationError, match=message):
        execute_selected_incentive_activation(
            network="mock",
            netuid=scope.netuid,
            intake_db=database,
            core_policy_path=paths[0],
            composition_policy_path=paths[1],
            approval_path=paths[2],
            expected_approval_digest=approval.digest,
            connect=lambda _network: _Subtensor(
                wrong_historical_hash=wrong_historical_hash
            ),
            read_finalized_head=lambda _subtensor: (
                101,
                "0x" + f"{101:064x}",
            ),
            fetch_metagraph=lambda _subtensor, netuid, *, block=None: type(
                "Metagraph",
                (),
                {
                    "netuid": netuid,
                    "block": 100 if block is None else block,
                    "block_hash": "0x" + f"{(100 if block is None else block):064x}",
                    "uids": [0, 1],
                    "hotkeys": ["validator", "reserve"],
                },
            )(),
        )


@pytest.mark.parametrize(
    ("retained_fault", "message"),
    [
        ("arena", "retained arena differs"),
        ("stack", "retained evaluation stack differs"),
        ("catalog", "retained catalog differs"),
    ],
)
def test_execute_activation_rejects_different_retained_campaign_authority(
    tmp_path: Path,
    retained_fault: str,
    message: str,
) -> None:
    scope, _core, _composition, approval, database, _arena, paths = _execution_paths(
        tmp_path, retained_fault=retained_fault
    )

    with pytest.raises(IncentiveActivationError, match=message):
        execute_selected_incentive_activation(
            network="mock",
            netuid=scope.netuid,
            intake_db=database,
            core_policy_path=paths[0],
            composition_policy_path=paths[1],
            approval_path=paths[2],
            expected_approval_digest=approval.digest,
            connect=lambda _network: _Subtensor(),
            read_finalized_head=lambda _subtensor: (
                101,
                "0x" + f"{101:064x}",
            ),
            fetch_metagraph=lambda _subtensor, netuid, *, block=None: type(
                "Metagraph",
                (),
                {
                    "netuid": netuid,
                    "block": 100 if block is None else block,
                    "block_hash": "0x" + f"{(100 if block is None else block):064x}",
                    "uids": [0, 1],
                    "hotkeys": ["validator", "reserve"],
                },
            )(),
        )


def test_execute_activation_rejects_different_registered_membership(
    tmp_path: Path,
) -> None:
    scope, _core, _composition, approval, database, _arena, paths = _execution_paths(
        tmp_path
    )

    with pytest.raises(IncentiveActivationError, match="membership digest differs"):
        execute_selected_incentive_activation(
            network="mock",
            netuid=scope.netuid,
            intake_db=database,
            core_policy_path=paths[0],
            composition_policy_path=paths[1],
            approval_path=paths[2],
            expected_approval_digest=approval.digest,
            connect=lambda _network: _Subtensor(),
            read_finalized_head=lambda _subtensor: (
                101,
                "0x" + f"{101:064x}",
            ),
            fetch_metagraph=lambda _subtensor, netuid, *, block=None: type(
                "Metagraph",
                (),
                {
                    "netuid": netuid,
                    "block": 100 if block is None else block,
                    "block_hash": "0x" + f"{(100 if block is None else block):064x}",
                    "uids": [0, 1, 2],
                    "hotkeys": ["validator", "reserve", "other-reserve"],
                },
            )(),
        )


def test_execute_activation_rejects_self_consistent_registered_reserve_change(
    tmp_path: Path,
) -> None:
    scope, core, composition, approval, database, _arena, paths = _execution_paths(
        tmp_path
    )
    changed_core = replace(core, reserve_hotkey="other-reserve")
    changed_composition = replace(
        composition, innovation_policy_digest=changed_core.digest
    )
    changed_approval = replace(
        approval,
        core_policy_digest=changed_core.digest,
        composition_policy_digest=changed_composition.digest,
        reserve_hotkey="other-reserve",
    )
    _write(paths[0], changed_core.to_dict())
    _write(paths[1], changed_composition.to_dict())
    _write(paths[2], changed_approval.to_dict())

    with pytest.raises(IncentiveActivationError, match="pinned approval"):
        execute_selected_incentive_activation(
            network="mock",
            netuid=scope.netuid,
            intake_db=database,
            core_policy_path=paths[0],
            composition_policy_path=paths[1],
            approval_path=paths[2],
            expected_approval_digest=approval.digest,
            connect=lambda _network: _Subtensor(),
            read_finalized_head=lambda _subtensor: (
                101,
                "0x" + f"{101:064x}",
            ),
            fetch_metagraph=lambda _subtensor, netuid, *, block=None: type(
                "Metagraph",
                (),
                {
                    "netuid": netuid,
                    "block": 100 if block is None else block,
                    "block_hash": "0x" + f"{(100 if block is None else block):064x}",
                    "uids": [0, 1, 2],
                    "hotkeys": ["validator", "reserve", "other-reserve"],
                },
            )(),
        )
