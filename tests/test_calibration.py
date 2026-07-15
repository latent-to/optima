from __future__ import annotations

from dataclasses import replace

import pytest

from optima.eval.calibration import (
    CALIBRATION_EVIDENCE_DOMAIN,
    CALIBRATION_EVIDENCE_SCHEMA,
    CalibrationContext,
    CalibrationControl,
    CalibrationEvidenceSet,
    CalibrationError,
    CalibrationManifest,
    CalibrationMeasurement,
    CalibrationObservation,
    CalibrationThresholdPolicy,
    MetricCalibration,
    SpeedCalibration,
    derive_calibration_manifest,
    publish_calibration_evidence,
    reopen_calibration_evidence,
)
from optima.eval.evidence_store import publish_evidence
from optima.stack_identity import canonical_json_bytes


def _digest(character: str) -> str:
    return character * 64


def _context(suffix: str = "") -> CalibrationContext:
    characters = "123456789abcdef"
    values = [character * 64 for character in characters]
    if suffix:
        values[10] = suffix * 64
    return CalibrationContext(*values[:11])


def _controls() -> tuple[CalibrationControl, ...]:
    return (
        CalibrationControl("negative", _digest("a"), _digest("b"), "FAIL"),
        CalibrationControl("positive", _digest("c"), _digest("d"), "PASS"),
        CalibrationControl("stock", _digest("e"), _digest("f"), "PASS"),
    )


def _manifest(*, status: str = "frozen") -> CalibrationManifest:
    return CalibrationManifest(
        context=_context(),
        algorithm_id="teacher-familywise-v1",
        status=status,
        speed=SpeedCalibration("0.005", "2", "0.1"),
        quality_metrics=(
            MetricCalibration("mean_nll", "lower", "0.02", "0.01"),
            MetricCalibration("task_score", "higher", "0.03", "0.02", "0.8"),
        ),
        familywise_z="2.576",
        raw_evidence_digest=_digest("1"),
        seed_digests=(_digest("2"), _digest("3")),
        controls=_controls(),
    )


def _observations() -> tuple[CalibrationObservation, ...]:
    rows = {
        "negative": (("0.4", "0.4"), ("0.2", "0.2"), ("0.4", "0.4"), ("0.4", "0.4")),
        "positive": (("0.001", "0.001"), ("0.01", "0.01"), ("0.001", "0.001"), ("0.9", "0.9")),
        "stock": (("0.001", "0.001"), ("0.001", "0.001"), ("0.001", "0.001"), ("0.92", "0.92")),
    }
    seeds = {"negative": _digest("7"), "positive": _digest("8"), "stock": _digest("9")}
    return tuple(
        CalibrationObservation(
            control_kind=kind,
            seed_digest=seeds[kind],
            measurements=(
                CalibrationMeasurement("mean_nll", rows[kind][0]),
                CalibrationMeasurement("speed_noise", rows[kind][1]),
                CalibrationMeasurement("task_score", rows[kind][2]),
                CalibrationMeasurement("task_score.absolute", rows[kind][3]),
            ),
        )
        for kind in ("negative", "positive", "stock")
    )


def _threshold_policy(*, status: str = "frozen") -> CalibrationThresholdPolicy:
    return CalibrationThresholdPolicy.from_manifest(_manifest(status=status))


def _with_measurement(
    observation: CalibrationObservation, name: str, values: tuple[str, ...]
) -> CalibrationObservation:
    return replace(
        observation,
        measurements=tuple(
            replace(row, values=values) if row.name == name else row
            for row in observation.measurements
        ),
    )


def _evidence(*, status: str = "frozen") -> CalibrationEvidenceSet:
    return CalibrationEvidenceSet.create(_threshold_policy(status=status), _observations())


def _derived_manifest(*, status: str = "frozen") -> CalibrationManifest:
    return derive_calibration_manifest(_threshold_policy(status=status), _observations())


def test_calibration_round_trip_and_digest_are_canonical() -> None:
    manifest = _manifest()
    reopened = CalibrationManifest.from_dict(manifest.to_dict())
    assert reopened == manifest
    assert reopened.digest == manifest.digest
    assert reopened.thresholds_frozen


def test_provisional_calibration_has_unfrozen_thresholds() -> None:
    manifest = _manifest(status="provisional")
    assert not manifest.thresholds_frozen
    assert CalibrationManifest.from_dict(manifest.to_dict()) == manifest


def test_zero_familywise_bound_cannot_freeze_uncertainty_away() -> None:
    with pytest.raises(CalibrationError, match="familywise_z"):
        replace(_manifest(), familywise_z="0")


def test_calibration_context_mismatch_rejects() -> None:
    manifest = _manifest()
    with pytest.raises(CalibrationError, match="stale or mismatched"):
        manifest.require_context(_context("e"))


@pytest.mark.parametrize(
    "field,value",
    (
        ("min_margin", "0"),
        ("min_margin", "0.010"),
        ("min_margin", "-0.1"),
        ("noise_multiplier", "0"),
        ("max_noise", "1"),
    ),
)
def test_speed_thresholds_reject_noncanonical_or_unsafe_values(field, value) -> None:
    values = {"min_margin": "0.005", "noise_multiplier": "2", "max_noise": "0.1"}
    values[field] = value
    with pytest.raises(CalibrationError):
        SpeedCalibration(**values)


def test_metric_policy_rejects_incompatible_floor() -> None:
    with pytest.raises(CalibrationError, match="cannot declare an absolute floor"):
        MetricCalibration("mean_nll", "lower", "0.1", "0.01", "0.9")


def test_metric_and_statistical_domains_are_bounded() -> None:
    with pytest.raises(CalibrationError, match="name/direction"):
        MetricCalibration("mean_nll", "higher", "0.1", "0.01")
    with pytest.raises(CalibrationError, match="must not exceed one"):
        MetricCalibration("task_score", "higher", "2", "0.1", "0.8")
    with pytest.raises(CalibrationError, match="statistical bound"):
        replace(_manifest(), familywise_z="999")


def test_controls_are_exact_and_outcomes_cannot_be_relabelled() -> None:
    with pytest.raises(CalibrationError, match="outcome disagrees"):
        CalibrationControl("negative", _digest("a"), _digest("b"), "PASS")
    with pytest.raises(CalibrationError, match="exactly negative, positive, stock"):
        replace(_manifest(), controls=tuple(reversed(_controls())))


def test_manifest_rejects_metric_and_seed_reordering() -> None:
    manifest = _manifest()
    with pytest.raises(CalibrationError, match="name-sorted"):
        replace(manifest, quality_metrics=tuple(reversed(manifest.quality_metrics)))
    with pytest.raises(CalibrationError, match="unique, and sorted"):
        replace(manifest, seed_digests=tuple(reversed(manifest.seed_digests)))


def test_strict_parser_rejects_unknown_or_tampered_fields() -> None:
    manifest = _manifest()
    raw = manifest.to_dict()
    raw["untrusted_threshold"] = "0"
    with pytest.raises(CalibrationError, match="fields mismatch"):
        CalibrationManifest.from_dict(raw)

    raw = manifest.to_dict()
    raw["context"]["arena_digest"] = _digest("f")
    reopened = CalibrationManifest.from_dict(raw)
    assert reopened.digest != manifest.digest
    with pytest.raises(CalibrationError, match="stale or mismatched"):
        reopened.require_context(manifest.context)


def test_all_zero_or_malformed_identity_rejects() -> None:
    values = _context().to_dict()
    values["arena_digest"] = "0" * 64
    with pytest.raises(CalibrationError, match="all-zero"):
        CalibrationContext.from_dict(values)
    values["arena_digest"] = "not-a-digest"
    with pytest.raises(CalibrationError, match="lowercase 64-hex"):
        CalibrationContext.from_dict(values)


def test_evidence_set_pure_rederives_every_manifest_evidence_identity() -> None:
    configured = _manifest()
    manifest = _derived_manifest()
    evidence = _evidence()

    assert manifest.raw_evidence_digest != configured.raw_evidence_digest
    assert manifest.seed_digests == tuple(
        sorted(row.seed_digest for row in _observations())
    )
    assert tuple(row.kind for row in manifest.controls) == ("negative", "positive", "stock")
    assert all(
        row.raw_evidence_digest not in {control.raw_evidence_digest for control in configured.controls}
        for row in manifest.controls
    )
    assert evidence.threshold_policy_digest == _threshold_policy().digest
    assert evidence.configured_manifest_digest == manifest.digest
    assert CalibrationEvidenceSet.from_dict(evidence.to_dict()) == evidence


def test_publish_reopen_rederives_and_binds_expected_manifest_and_context(tmp_path) -> None:
    policy, manifest = _threshold_policy(), _derived_manifest()
    evidence = _evidence()
    reference = publish_calibration_evidence(tmp_path / "evidence", evidence)

    reopened = reopen_calibration_evidence(
        tmp_path / "evidence",
        reference,
        expected_threshold_policy=policy,
        expected_manifest=manifest,
        expected_context=manifest.context,
    )
    assert reopened == manifest


def test_provisional_evidence_reopens_but_is_never_crown_authoritative(tmp_path) -> None:
    policy = _threshold_policy(status="provisional")
    manifest = _derived_manifest(status="provisional")
    evidence = _evidence(status="provisional")
    reference = publish_calibration_evidence(tmp_path / "evidence", evidence)
    reopened = reopen_calibration_evidence(
        tmp_path / "evidence",
        reference,
        expected_threshold_policy=policy,
        expected_manifest=manifest,
        expected_context=manifest.context,
    )
    assert not reopened.thresholds_frozen


def test_raw_value_tamper_cannot_hide_behind_the_aggregate_digest(tmp_path) -> None:
    policy, manifest = _threshold_policy(), _derived_manifest()
    evidence = _evidence()
    raw = evidence.to_dict()
    raw["observations"][0]["measurements"][0]["values"][0] = "0.6"
    root = tmp_path / "evidence"
    reference = publish_evidence(
        root,
        canonical_json_bytes(raw),
        domain=CALIBRATION_EVIDENCE_DOMAIN,
        media_type="application/json",
        schema=CALIBRATION_EVIDENCE_SCHEMA,
    )
    with pytest.raises(CalibrationError, match="another configured manifest"):
        reopen_calibration_evidence(
            root,
            reference,
            expected_threshold_policy=policy,
            expected_manifest=manifest,
            expected_context=manifest.context,
        )


def test_reopen_rejects_threshold_relabel_even_when_raw_identities_are_self_consistent(tmp_path) -> None:
    expected_policy, expected_manifest = _threshold_policy(), _derived_manifest()
    relabelled_policy = replace(
        expected_policy, speed=SpeedCalibration("0.006", "2", "0.1")
    )
    relabelled = CalibrationEvidenceSet.create(
        relabelled_policy,
        _observations(),
    )
    reference = publish_calibration_evidence(tmp_path / "evidence", relabelled)
    with pytest.raises(CalibrationError, match="another validator threshold policy"):
        reopen_calibration_evidence(
            tmp_path / "evidence",
            reference,
            expected_threshold_policy=expected_policy,
            expected_manifest=expected_manifest,
            expected_context=expected_manifest.context,
        )


@pytest.mark.parametrize(
    "rows",
    (
        _observations()[:-1],
        tuple(reversed(_observations())),
        (_observations()[0], _observations()[0], _observations()[2]),
    ),
)
def test_evidence_controls_are_exact_and_ordered(rows) -> None:
    with pytest.raises(CalibrationError, match="exactly negative, positive, stock"):
        CalibrationEvidenceSet.create(_threshold_policy(), rows)


def test_observations_require_distinct_seeds_and_exact_metric_coverage() -> None:
    rows = list(_observations())
    rows[1] = replace(rows[1], seed_digest=rows[0].seed_digest)
    with pytest.raises(CalibrationError, match="distinct seeds"):
        CalibrationEvidenceSet.create(_threshold_policy(), rows)

    rows = list(_observations())
    rows[1] = replace(rows[1], measurements=rows[1].measurements[1:])
    with pytest.raises(CalibrationError, match="different metric sets"):
        CalibrationEvidenceSet.create(_threshold_policy(), rows)

    rows = tuple(
        replace(row, measurements=tuple(item for item in row.measurements if item.name != "speed_noise"))
        for row in _observations()
    )
    with pytest.raises(CalibrationError, match="do not exactly cover"):
        CalibrationEvidenceSet.create(_threshold_policy(), rows)


@pytest.mark.parametrize(
    "index,measurement,values,kind",
    (
        (2, "speed_noise", ("0.2", "0.2"), "stock"),
        (1, "mean_nll", ("0.2", "0.2"), "positive"),
        (1, "task_score.absolute", ("0.2", "0.2"), "positive"),
    ),
)
def test_registered_control_grader_rejects_contradictory_passes(
    index, measurement, values, kind
) -> None:
    rows = list(_observations())
    rows[index] = _with_measurement(rows[index], measurement, values)
    with pytest.raises(CalibrationError, match=rf"{kind} control outcome contradicts"):
        CalibrationEvidenceSet.create(_threshold_policy(), rows)


def test_registered_control_grader_requires_negative_to_demonstrably_fail() -> None:
    rows = list(_observations())
    passing = _observations()[1]
    rows[0] = replace(rows[0], measurements=passing.measurements)
    with pytest.raises(CalibrationError, match="negative control outcome contradicts"):
        CalibrationEvidenceSet.create(_threshold_policy(), rows)


def test_threshold_policy_rejects_unregistered_control_algorithm() -> None:
    with pytest.raises(CalibrationError, match="unknown algorithm"):
        replace(_threshold_policy(), algorithm_id="invented-policy-v1")


def test_reopen_rejects_wrong_context_and_wrong_artifact_type(tmp_path) -> None:
    policy, manifest = _threshold_policy(), _derived_manifest()
    evidence = _evidence()
    root = tmp_path / "evidence"
    reference = publish_calibration_evidence(root, evidence)
    with pytest.raises(CalibrationError, match="stale or mismatched"):
        reopen_calibration_evidence(
            root,
            reference,
            expected_threshold_policy=policy,
            expected_manifest=manifest,
            expected_context=_context("e"),
        )

    wrong = publish_evidence(
        root,
        canonical_json_bytes(evidence.to_dict()),
        domain="other-calibration",
        media_type="application/json",
        schema=CALIBRATION_EVIDENCE_SCHEMA,
    )
    with pytest.raises(CalibrationError, match="type is not authoritative"):
        reopen_calibration_evidence(
            root,
            wrong,
            expected_threshold_policy=policy,
            expected_manifest=manifest,
            expected_context=manifest.context,
        )


def test_reopen_rejects_noncanonical_and_float_json(tmp_path) -> None:
    policy, manifest = _threshold_policy(), _derived_manifest()
    evidence = _evidence()
    root = tmp_path / "evidence"
    payload = canonical_json_bytes(evidence.to_dict())
    noncanonical = publish_evidence(
        root,
        payload + b"\n",
        domain=CALIBRATION_EVIDENCE_DOMAIN,
        media_type="application/json",
        schema=CALIBRATION_EVIDENCE_SCHEMA,
    )
    with pytest.raises(CalibrationError, match="not canonically encoded"):
        reopen_calibration_evidence(
            root,
            noncanonical,
            expected_threshold_policy=policy,
            expected_manifest=manifest,
            expected_context=manifest.context,
        )

    float_payload = payload.replace(b'"0.01"', b"0.01", 1)
    floating = publish_evidence(
        root,
        float_payload,
        domain=CALIBRATION_EVIDENCE_DOMAIN,
        media_type="application/json",
        schema=CALIBRATION_EVIDENCE_SCHEMA,
    )
    with pytest.raises(CalibrationError, match="contains a float"):
        reopen_calibration_evidence(
            root,
            floating,
            expected_threshold_policy=policy,
            expected_manifest=manifest,
            expected_context=manifest.context,
        )
