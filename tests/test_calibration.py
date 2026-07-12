from __future__ import annotations

from dataclasses import replace

import pytest

from optima.eval.calibration import (
    CalibrationContext,
    CalibrationControl,
    CalibrationError,
    CalibrationManifest,
    MetricCalibration,
    SpeedCalibration,
)


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
    with pytest.raises(CalibrationError, match="fields do not match"):
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
