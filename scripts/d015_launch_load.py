#!/usr/bin/env python3
"""Replay the tracked D-015 one-campaign load and rental-ROI sensitivity.

The original D-015 selection tape rotated one aggregate win among target
families.  This supplement deliberately gives every active family an
independent CROWN stream.  It uses the production claim and composed-epoch
arithmetic; the only synthetic inputs are win cadence and discovery load.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
)
from optima.stack_identity import (
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
    sha256_hex,
)


SCHEMA = "optima.incentive-launch-load.d015.v1"
REPORT_SCHEMA = "optima.incentive-launch-load-report.d015.v1"


class LaunchLoadError(ValueError):
    """The tracked load configuration or generated evidence is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchLoadError(message)


def _exact_dict(value: object, fields: set[str], label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact JSON object")
    row = dict(value)
    _require(set(row) == fields, f"{label} fields differ")
    return row


def _positive_ints(value: object, label: str) -> tuple[int, ...]:
    _require(type(value) is list and bool(value), f"{label} must be nonempty")
    rows = tuple(value)
    _require(
        all(type(item) is int and item > 0 for item in rows),
        f"{label} must contain positive integers",
    )
    _require(rows == tuple(sorted(set(rows))), f"{label} must be unique and sorted")
    return rows


def _speedup(value: object, label: str) -> str:
    _require(
        isinstance(value, str)
        and value.startswith("1.")
        and value[2:].isdigit()
        and not value.endswith("0"),
        f"{label} must be a canonical gross speedup string",
    )
    return value


def load_config(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LaunchLoadError("configuration is not UTF-8 JSON") from exc
    config = _exact_dict(
        value,
        {
            "anchor_speedups",
            "campaign_label",
            "matrix",
            "nonclaims",
            "payout_cases",
            "policy",
            "roi",
            "same_day_burst_claim_counts",
            "schema",
        },
        "configuration",
    )
    _require(config["schema"] == SCHEMA, "configuration schema differs")
    _require(
        isinstance(config["campaign_label"], str) and bool(config["campaign_label"]),
        "campaign_label is malformed",
    )
    _require(
        type(config["nonclaims"]) is list
        and bool(config["nonclaims"])
        and all(isinstance(item, str) and item for item in config["nonclaims"]),
        "nonclaims are malformed",
    )
    anchors = config["anchor_speedups"]
    _require(type(anchors) is list and bool(anchors), "anchor_speedups are malformed")
    for index, value in enumerate(anchors):
        _speedup(value, f"anchor_speedups[{index}]")

    policy = _exact_dict(
        config["policy"],
        {
            "beta_ppm",
            "claim_lifetime_blocks",
            "clock_reset_threshold_log_units_ppm",
            "core_selection_report_digest",
            "discovery_cap_units",
            "discovery_selection_report_digest",
            "epoch_blocks",
            "k_ppm",
            "reserve_ppm",
            "tau_blocks",
        },
        "policy",
    )
    _require(policy["reserve_ppm"] == 100_000, "reserve_ppm differs")
    _require(policy["epoch_blocks"] == 7_200, "epoch_blocks differs")
    _require(policy["beta_ppm"] == 100_000, "beta_ppm differs")
    _require(policy["tau_blocks"] == 648_000, "tau_blocks differs")
    _require(
        policy["claim_lifetime_blocks"] == 648_000,
        "claim_lifetime_blocks differs",
    )
    _require(policy["k_ppm"] == PPM, "k_ppm differs")
    _require(
        policy["clock_reset_threshold_log_units_ppm"] == 1,
        "clock reset threshold differs",
    )
    _require(policy["discovery_cap_units"] == 50_000, "discovery cap differs")
    for field in (
        "core_selection_report_digest",
        "discovery_selection_report_digest",
    ):
        require_sha256_hex(policy[field], field=field)

    matrix = _exact_dict(
        config["matrix"],
        {
            "active_family_streams",
            "arrival_windows_days",
            "cadence_days",
            "speedup",
        },
        "matrix",
    )
    _require(
        _positive_ints(matrix["active_family_streams"], "active_family_streams")
        == (1, 2, 5, 10),
        "active-family matrix differs",
    )
    _require(
        _positive_ints(matrix["cadence_days"], "cadence_days") == (7, 14, 30, 90),
        "cadence matrix differs",
    )
    _require(
        _positive_ints(matrix["arrival_windows_days"], "arrival_windows_days")
        == (14, 365),
        "arrival windows differ",
    )
    _speedup(matrix["speedup"], "matrix speedup")

    cases = config["payout_cases"]
    _require(type(cases) is list and len(cases) == 2, "payout cases differ")
    parsed_cases = [
        _exact_dict(case, {"discovery_units_per_epoch", "id"}, "payout case")
        for case in cases
    ]
    _require(
        [(row["id"], row["discovery_units_per_epoch"]) for row in parsed_cases]
        == [("empty-discovery", 0), ("saturated-discovery", 50_000)],
        "payout cases differ",
    )
    _require(
        _positive_ints(config["same_day_burst_claim_counts"], "burst counts")
        == (19, 20),
        "burst controls differ",
    )
    roi = _exact_dict(
        config["roi"],
        {"campaign_cost_usd_cents", "success_probability_ppm"},
        "roi",
    )
    _require(
        _positive_ints(roi["campaign_cost_usd_cents"], "campaign costs")
        == (100_000, 150_000),
        "campaign costs differ",
    )
    _require(roi["success_probability_ppm"] == 250_000, "success probability differs")
    canonical_json_bytes(config)
    return config


def _digest(label: str) -> str:
    return sha256_hex(label.encode("utf-8"))


def _campaign_id(config: dict[str, Any]) -> str:
    return _digest(f"d015-campaign:{config['campaign_label']}")


def _family_id(config: dict[str, Any], index: int) -> str:
    return _digest(f"d015-family:{config['campaign_label']}:{index:03d}")


def _innovation_policy(
    config: dict[str, Any], family_count: int
) -> FiniteDebtPolicyManifest:
    policy = config["policy"]
    campaign_id = _campaign_id(config)
    return FiniteDebtPolicyManifest(
        campaign_budget_shares=(CampaignBudgetShare(campaign_id, PPM),),
        reward_family_campaigns=tuple(
            RewardFamilyCampaign(_family_id(config, index), campaign_id)
            for index in range(family_count)
        ),
        selection_report_digest=policy["core_selection_report_digest"],
        reserve_hotkey="d015-load-reserve",
        reserve_ppm=policy["reserve_ppm"],
        epoch_blocks=policy["epoch_blocks"],
        beta_ppm=policy["beta_ppm"],
        tau_blocks=policy["tau_blocks"],
        lifetime_blocks=policy["claim_lifetime_blocks"],
        k_ppm=policy["k_ppm"],
        improvement_basis=IMPROVEMENT_GROSS,
        clock_reset_threshold_log_units_ppm=policy[
            "clock_reset_threshold_log_units_ppm"
        ],
    )


def _composition_policy(
    config: dict[str, Any], innovation: FiniteDebtPolicyManifest
) -> IncentiveCompositionPolicyManifest:
    policy = config["policy"]
    return IncentiveCompositionPolicyManifest(
        innovation_policy_digest=innovation.digest,
        selection_report_digest=policy["discovery_selection_report_digest"],
        reserve_ppm=policy["reserve_ppm"],
        epoch_blocks=policy["epoch_blocks"],
        discovery_cap_units=policy["discovery_cap_units"],
        per_award_principal_cap_epochs=1,
        discovery_lifetime_blocks=policy["claim_lifetime_blocks"],
    )


def _issue_crown(
    config: dict[str, Any],
    policy: FiniteDebtPolicyManifest,
    *,
    family_index: int,
    event_index: int,
    day: int,
    speedup: str,
    prior_day: int | None,
) -> DebtClaimState:
    epoch_blocks = policy.epoch_blocks
    block = day * epoch_blocks
    prior_block = None if prior_day is None else prior_day * epoch_blocks
    label = f"family:{family_index}:event:{event_index}:day:{day}:speed:{speedup}"
    claim = issue_innovation_claim(
        policy,
        family_id=_family_id(config, family_index),
        candidate_digest=_digest(f"d015-candidate:{label}"),
        retained_evidence_digest=_digest(f"d015-evidence:{label}"),
        hotkey=f"d015-miner-{family_index:03d}",
        settled_speedup=speedup,
        threshold_speedup="1",
        accepted_crown_block=block,
        prior_accepted_crown_block=prior_block,
        settlement_block=block,
    )
    return DebtClaimState(claim, DebtClaimBalance.open(claim))


def _event_tape(
    *, family_count: int, cadence_days: int, arrival_days: int
) -> tuple[tuple[int, int, int, int | None], ...]:
    events: list[tuple[int, int, int, int | None]] = []
    last_by_family: dict[int, int] = {}
    event_index = 0
    for day in range(0, arrival_days, cadence_days):
        for family_index in range(family_count):
            events.append(
                (day, family_index, event_index, last_by_family.get(family_index))
            )
            last_by_family[family_index] = day
            event_index += 1
    return tuple(events)


def _pro_rata(open_rows: list[dict[str, Any]], capacity: int) -> None:
    """Apply the production claim-digest largest-remainder rule in place."""

    total = sum(row["remaining_units"] for row in open_rows)
    payout = min(capacity, total)
    if payout == 0:
        return
    amounts: dict[str, int] = {}
    if total <= capacity:
        amounts = {row["claim_id"]: row["remaining_units"] for row in open_rows}
    else:
        remainders: list[tuple[int, str]] = []
        for row in open_rows:
            quotient, remainder = divmod(row["remaining_units"] * payout, total)
            amounts[row["claim_id"]] = quotient
            remainders.append((remainder, row["claim_id"]))
        missing = payout - sum(amounts.values())
        for _remainder, claim_id in sorted(
            remainders, key=lambda item: (-item[0], item[1])
        )[:missing]:
            amounts[claim_id] += 1
    for row in open_rows:
        amount = amounts[row["claim_id"]]
        row["paid_units"] += amount
        row["remaining_units"] -= amount
        if row["remaining_units"] == 0:
            row["status"] = "paid"
            row["final_paid_day"] = row["current_day"]


def _run_tape(
    config: dict[str, Any],
    *,
    family_count: int,
    events: tuple[tuple[int, int, int, int | None], ...],
    speedup: str,
    discovery_units_per_epoch: int,
) -> dict[str, int | None]:
    _require(bool(events), "simulation tape is empty")
    innovation_policy = _innovation_policy(config, family_count)
    composition_policy = _composition_policy(config, innovation_policy)
    _require(
        discovery_units_per_epoch in (0, composition_policy.discovery_cap_units),
        "simulation discovery load must be empty or saturated",
    )
    by_day: dict[int, list[tuple[int, int, int | None]]] = {}
    for day, family_index, event_index, prior_day in events:
        by_day.setdefault(day, []).append((family_index, event_index, prior_day))

    # Claim issuance is taken from the production implementation. Payout state is
    # kept as small JSON rows so the 64-cell matrix remains a fast CI check; the
    # allocator below is the same digest-ordered integer largest-remainder rule.
    first_principal_by_family = {
        family_index: _issue_crown(
            config,
            innovation_policy,
            family_index=family_index,
            event_index=family_index,
            day=0,
            speedup=speedup,
            prior_day=None,
        ).claim.principal_units
        for family_index in range(family_count)
    }
    recurring_principal_by_gap: dict[int, int] = {}
    for day, family_index, event_index, prior_day in events:
        if prior_day is not None and day - prior_day not in recurring_principal_by_gap:
            recurring_principal_by_gap[day - prior_day] = _issue_crown(
                config,
                innovation_policy,
                family_index=family_index,
                event_index=event_index,
                day=day,
                speedup=speedup,
                prior_day=prior_day,
            ).claim.principal_units

    states: list[dict[str, Any]] = []
    maximum_open_units = 0
    maximum_open_claims = 0
    last_issue_day = max(by_day)
    lifetime_days, remainder = divmod(
        innovation_policy.lifetime_blocks, innovation_policy.epoch_blocks
    )
    _require(remainder == 0, "claim lifetime is not an integer number of epochs")
    for day in range(last_issue_day + lifetime_days + 1):
        for state in states:
            if state["status"] == "open" and day >= state["expires_day"]:
                state["expired_units"] = state["remaining_units"]
                state["remaining_units"] = 0
                state["status"] = "expired"
        for family_index, event_index, prior_day in by_day.get(day, ()):
            principal = (
                first_principal_by_family[family_index]
                if prior_day is None
                else recurring_principal_by_gap[day - prior_day]
            )
            claim_id = canonical_digest(
                "optima.incentives.d015-load-claim",
                {
                    "day": day,
                    "event_index": event_index,
                    "family_index": family_index,
                    "principal_units": principal,
                    "speedup": speedup,
                },
            )
            states.append(
                {
                    "claim_id": claim_id,
                    "current_day": day,
                    "expired_units": 0,
                    "expires_day": day + lifetime_days,
                    "final_paid_day": None,
                    "issue_day": day,
                    "paid_units": 0,
                    "principal_units": principal,
                    "remaining_units": principal,
                    "status": "open",
                }
            )
        open_rows = [state for state in states if state["status"] == "open"]
        maximum_open_units = max(
            maximum_open_units,
            sum(state["remaining_units"] for state in open_rows),
        )
        maximum_open_claims = max(maximum_open_claims, len(open_rows))
        for state in open_rows:
            state["current_day"] = day
        _pro_rata(
            open_rows,
            PPM - innovation_policy.reserve_ppm - discovery_units_per_epoch,
        )

    _require(
        all(state["status"] != "open" for state in states),
        "simulation ended with open claims",
    )
    principal = sum(state["principal_units"] for state in states)
    paid = sum(state["paid_units"] for state in states)
    expired = sum(state["expired_units"] for state in states)
    _require(principal == paid + expired, "simulation did not conserve principal")
    paid_latencies = [
        state["final_paid_day"] - state["issue_day"]
        for state in states
        if state["status"] == "paid" and state["final_paid_day"] is not None
    ]
    return {
        "claim_count": len(states),
        "expired_claim_count": sum(
            state["status"] == "expired" for state in states
        ),
        "expired_units": expired,
        "fully_paid_claim_count": len(paid_latencies),
        "maximum_open_claims": maximum_open_claims,
        "maximum_open_units": maximum_open_units,
        "maximum_paid_claim_latency_days": max(paid_latencies, default=None),
        "paid_fraction_ppm": paid * PPM // principal,
        "paid_units": paid,
        "principal_units": principal,
    }


def _ceil_div(numerator: int, denominator: int) -> int:
    _require(denominator > 0, "ROI denominator must be positive")
    return (numerator + denominator - 1) // denominator


def _roi(
    config: dict[str, Any], *, claim_principal_units: int, collection_ppm: int
) -> dict[str, Any]:
    roi = config["roi"]
    probability = roi["success_probability_ppm"]
    denominator = probability * collection_ppm * claim_principal_units
    return {
        "break_even_vector_day_usd_cents": [
            _ceil_div(cost * PPM**3, denominator)
            for cost in roi["campaign_cost_usd_cents"]
        ],
        "campaign_cost_usd_cents": list(roi["campaign_cost_usd_cents"]),
        "claim_principal_units": claim_principal_units,
        "collection_ppm": collection_ppm,
        "success_probability_ppm": probability,
    }


def _anchor_claims(config: dict[str, Any]) -> list[dict[str, Any]]:
    policy = _innovation_policy(config, 1)
    rows = []
    for index, speedup in enumerate(config["anchor_speedups"]):
        state = _issue_crown(
            config,
            policy,
            family_index=0,
            event_index=index,
            day=0,
            speedup=speedup,
            prior_day=None,
        )
        rows.append(
            {
                "log_units_ppm": state.claim.log_units_ppm,
                "principal_units": state.claim.principal_units,
                "speedup": speedup,
                "time_multiplier_ppm": state.claim.time_multiplier_ppm,
            }
        )
    return rows


def simulate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    anchors = _anchor_claims(config)
    anchor_by_speed = {row["speedup"]: row for row in anchors}
    matrix = config["matrix"]
    matrix_rows: list[dict[str, Any]] = []
    for arrival_days in matrix["arrival_windows_days"]:
        window = "launch-14-day" if arrival_days == 14 else "sustained-365-day"
        for active_families in matrix["active_family_streams"]:
            for cadence_days in matrix["cadence_days"]:
                events = _event_tape(
                    family_count=active_families,
                    cadence_days=cadence_days,
                    arrival_days=arrival_days,
                )
                for payout_case in config["payout_cases"]:
                    result = _run_tape(
                        config,
                        family_count=active_families,
                        events=events,
                        speedup=matrix["speedup"],
                        discovery_units_per_epoch=payout_case[
                            "discovery_units_per_epoch"
                        ],
                    )
                    matrix_rows.append(
                        {
                            "active_family_streams": active_families,
                            "arrival_days": arrival_days,
                            "cadence_days": cadence_days,
                            "discovery_units_per_epoch": payout_case[
                                "discovery_units_per_epoch"
                            ],
                            "payout_case": payout_case["id"],
                            "roi_25pct_first_claim": _roi(
                                config,
                                claim_principal_units=anchor_by_speed[
                                    matrix["speedup"]
                                ]["principal_units"],
                                collection_ppm=result["paid_fraction_ppm"],
                            ),
                            "speedup": matrix["speedup"],
                            "window": window,
                            **result,
                        }
                    )

    burst_rows: list[dict[str, Any]] = []
    for claim_count in config["same_day_burst_claim_counts"]:
        events = tuple((0, index, index, None) for index in range(claim_count))
        for payout_case in config["payout_cases"]:
            burst_rows.append(
                {
                    "claim_count_control": claim_count,
                    "discovery_units_per_epoch": payout_case[
                        "discovery_units_per_epoch"
                    ],
                    "payout_case": payout_case["id"],
                    "speedup": matrix["speedup"],
                    **_run_tape(
                        config,
                        family_count=claim_count,
                        events=events,
                        speedup=matrix["speedup"],
                        discovery_units_per_epoch=payout_case[
                            "discovery_units_per_epoch"
                        ],
                    ),
                }
            )

    launch_weekly_saturated = [
        row
        for row in matrix_rows
        if row["window"] == "launch-14-day"
        and row["cadence_days"] == 7
        and row["payout_case"] == "saturated-discovery"
    ]
    sustained_weekly_saturated = [
        row
        for row in matrix_rows
        if row["window"] == "sustained-365-day"
        if row["cadence_days"] == 7
        and row["payout_case"] == "saturated-discovery"
    ]
    source_path = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "anchor_claims": anchors,
        "burst_rows": burst_rows,
        "config_digest": canonical_digest("optima.incentives.d015-load-config", config),
        "config_sha256": sha256_hex(config_path.read_bytes()),
        "matrix_rows": matrix_rows,
        "nonclaims": list(config["nonclaims"]),
        "schema": REPORT_SCHEMA,
        "source_sha256": sha256_hex(source_path.read_bytes()),
        "summary": {
            "matrix_row_count": len(matrix_rows),
            "minimum_paid_fraction_ppm": min(
                row["paid_fraction_ppm"] for row in matrix_rows
            ),
            "launch_weekly_saturated_discovery": [
                {
                    "active_family_streams": row["active_family_streams"],
                    "break_even_vector_day_usd_cents": row[
                        "roi_25pct_first_claim"
                    ]["break_even_vector_day_usd_cents"],
                    "expired_units": row["expired_units"],
                    "paid_fraction_ppm": row["paid_fraction_ppm"],
                }
                for row in launch_weekly_saturated
            ],
            "sustained_weekly_saturated_discovery": [
                {
                    "active_family_streams": row["active_family_streams"],
                    "break_even_vector_day_usd_cents": row[
                        "roi_25pct_first_claim"
                    ]["break_even_vector_day_usd_cents"],
                    "expired_units": row["expired_units"],
                    "paid_fraction_ppm": row["paid_fraction_ppm"],
                }
                for row in sustained_weekly_saturated
            ],
        },
    }
    payload["report_digest"] = canonical_digest(
        "optima.incentives.d015-load-report", payload
    )
    canonical_json_bytes(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    report = simulate(arguments.config)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_bytes(canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
