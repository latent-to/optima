from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from optima.finite_debt import (
    IMPROVEMENT_GROSS,
    PPM,
    CampaignBudgetShare,
    DebtClaimBalance,
    DebtClaimState,
    FiniteDebtPolicyManifest,
    RewardFamilyCampaign,
    issue_innovation_claim,
)
from optima.incentive_composition import (
    IncentiveCompositionPolicyManifest,
    project_composed_epoch,
)
from optima.stack_identity import canonical_digest, canonical_json_bytes, sha256_hex
from scripts import d015_launch_load as d015


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "evidence" / "incentives" / "d015_launch_load_config.json"
REPORT = ROOT / "evidence" / "incentives" / "D015_LAUNCH_LOAD_REPORT.md"


def _digest(label: str) -> str:
    return sha256_hex(label.encode())


@pytest.fixture(scope="module")
def replayed_report() -> dict[str, object]:
    return d015.simulate(CONFIG)


def test_tracked_d015_launch_load_report_replays_exactly(
    replayed_report: dict[str, object],
) -> None:
    assert canonical_json_bytes(json.loads(CONFIG.read_bytes())) + b"\n" == CONFIG.read_bytes()
    markdown = REPORT.read_text(encoding="utf-8")
    match = re.search(r"Semantic report digest: `([0-9a-f]{64})`", markdown)
    assert match is not None
    assert match.group(1) == replayed_report["report_digest"]
    unsigned = dict(replayed_report)
    digest = unsigned.pop("report_digest")
    assert digest == canonical_digest("optima.incentives.d015-load-report", unsigned)


def test_d015_matrix_models_independent_active_family_streams(
    replayed_report: dict[str, object],
) -> None:
    rows = replayed_report["matrix_rows"]
    assert len(rows) == 2 * 4 * 4 * 2
    assert {
        (row["window"], row["active_family_streams"], row["cadence_days"])
        for row in rows
    } == {
        (window, families, cadence)
        for window in ("launch-14-day", "sustained-365-day")
        for families in (1, 2, 5, 10)
        for cadence in (7, 14, 30, 90)
    }
    launch_weekly_saturated = {
        row["active_family_streams"]: row
        for row in rows
        if row["window"] == "launch-14-day"
        and row["cadence_days"] == 7
        and row["payout_case"] == "saturated-discovery"
    }
    assert launch_weekly_saturated[1]["paid_fraction_ppm"] == 1_000_000
    assert launch_weekly_saturated[2]["paid_fraction_ppm"] == 1_000_000
    assert launch_weekly_saturated[5]["paid_fraction_ppm"] == 1_000_000
    assert launch_weekly_saturated[10]["paid_fraction_ppm"] == 990_211
    assert launch_weekly_saturated[10]["expired_units"] == 765_240

    sustained_weekly_saturated = {
        row["active_family_streams"]: row
        for row in rows
        if row["window"] == "sustained-365-day"
        and row["cadence_days"] == 7
        and row["payout_case"] == "saturated-discovery"
    }
    assert sustained_weekly_saturated[1]["paid_fraction_ppm"] == 1_000_000
    assert sustained_weekly_saturated[2]["paid_fraction_ppm"] == 856_341
    assert sustained_weekly_saturated[5]["paid_fraction_ppm"] == 371_198
    assert sustained_weekly_saturated[10]["paid_fraction_ppm"] == 185_635
    assert sustained_weekly_saturated[10]["expired_units"] == 1_692_903_490


def test_d015_burst_controls_cover_empty_and_saturated_discovery(
    replayed_report: dict[str, object],
) -> None:
    rows = {
        (row["claim_count_control"], row["payout_case"]): row
        for row in replayed_report["burst_rows"]
    }
    assert rows[(19, "saturated-discovery")]["expired_units"] == 0
    assert rows[(20, "saturated-discovery")]["expired_units"] == 1_393_940
    assert rows[(20, "empty-discovery")]["expired_units"] == 0


def test_two_campaign_claim_sizing_is_not_a_hard_payout_silo() -> None:
    campaign_a = _digest("campaign-a")
    campaign_b = _digest("campaign-b")
    family_a = _digest("family-a")
    family_b = _digest("family-b")
    policy = FiniteDebtPolicyManifest(
        campaign_budget_shares=(
            CampaignBudgetShare(campaign_a, 500_000),
            CampaignBudgetShare(campaign_b, 500_000),
        ),
        reward_family_campaigns=(
            RewardFamilyCampaign(family_a, campaign_a),
            RewardFamilyCampaign(family_b, campaign_b),
        ),
        selection_report_digest=_digest("d015-report"),
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
        innovation_policy_digest=policy.digest,
        selection_report_digest=_digest("d013-report"),
        reserve_ppm=100_000,
        epoch_blocks=7_200,
        discovery_cap_units=50_000,
        per_award_principal_cap_epochs=1,
        discovery_lifetime_blocks=648_000,
    )
    claim = issue_innovation_claim(
        policy,
        family_id=family_a,
        candidate_digest=_digest("candidate"),
        retained_evidence_digest=_digest("evidence"),
        hotkey="miner-a",
        settled_speedup="1.05",
        threshold_speedup="1",
        accepted_crown_block=0,
        prior_accepted_crown_block=None,
        settlement_block=0,
    )
    assert claim.reference_campaign_pool_units == 450_000
    assert claim.principal_units == 2_206_516
    projection = project_composed_epoch(
        policy,
        composition,
        effective_block=0,
        innovation_states=(DebtClaimState(claim, DebtClaimBalance.open(claim)),),
        discovery_states=(),
    )
    assert projection.innovation_capacity_units == 900_000
    assert projection.innovation_payout_units == 900_000
    assert projection.innovation_allocations[0].units == 900_000
