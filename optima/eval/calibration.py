"""Content-addressed calibration authority for qualification gates.

Calibration is policy, not a convenient collection of local floats.  A frozen
manifest binds the derivation algorithm, pristine reference, exact arena/runtime/
hardware/workload context, verifier policy, raw evidence, seeds, and controls.
Qualification may use its thresholds only when every binding reopens exactly.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from optima.stack_identity import (
    StackIdentityError,
    canonical_digest,
    require_sha256_hex,
)


CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_POLICY_VERSION = "qualification-calibration.v1"
_NAME = re.compile(r"[a-z][a-z0-9._-]*\Z")
_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?\Z")
_STATUSES = frozenset({"frozen", "provisional"})
_CONTROL_OUTCOMES = {"stock": "PASS", "positive": "PASS", "negative": "FAIL"}
_METRIC_DIRECTIONS = {
    "argmax_rate": "lower",
    "coverage_dev": "lower",
    "mean_nll": "lower",
    "tail_rate": "lower",
    "task_score": "higher",
    "topk_kl": "lower",
    "worst_nll": "lower",
}
_UNIT_METRICS = frozenset({"argmax_rate", "coverage_dev", "tail_rate", "task_score"})


class CalibrationError(ValueError):
    """A calibration manifest is malformed, stale, or not crown-authoritative."""


def _digest(value: object, *, field: str) -> str:
    try:
        result = require_sha256_hex(value, field=field)
    except StackIdentityError as exc:
        raise CalibrationError(str(exc)) from exc
    if result == "0" * 64:
        raise CalibrationError(f"{field} must not be the all-zero digest")
    return result


def _name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise CalibrationError(f"{field} must be a canonical policy name")
    return value


def _decimal(value: object, *, field: str, positive: bool = False) -> str:
    if not isinstance(value, str) or _DECIMAL.fullmatch(value) is None:
        raise CalibrationError(f"{field} must be a canonical nonnegative decimal string")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise CalibrationError(f"{field} is not a decimal") from exc
    if not number.is_finite() or number < 0 or (positive and number <= 0):
        raise CalibrationError(f"{field} is outside its allowed range")
    normalized = format(number, "f").rstrip("0").rstrip(".") if "." in value else value
    if normalized == "":
        normalized = "0"
    if normalized != value:
        raise CalibrationError(f"{field} is not canonically encoded")
    return value


def decimal_value(value: str) -> Decimal:
    """Convert one already-validated policy decimal for pure scoring math."""

    return Decimal(_decimal(value, field="policy decimal"))


def _strict(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or frozenset(value) != fields:
        raise CalibrationError(f"{label} fields do not match the schema")
    return value


@dataclass(frozen=True)
class CalibrationContext:
    reference_manifest_digest: str
    arena_digest: str
    runtime_digest: str
    base_engine_digest: str
    model_revision_digest: str
    model_manifest_digest: str
    model_content_digest: str
    logical_hardware_digest: str
    workload_digest: str
    verification_policy_digest: str
    controller_distribution_digest: str

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))

    def to_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> "CalibrationContext":
        fields = frozenset(cls.__dataclass_fields__)
        raw = _strict(value, fields, label="calibration context")
        return cls(**{field: raw[field] for field in fields})

    @property
    def digest(self) -> str:
        return canonical_digest("optima.qualification.calibration-context", self.to_dict())


@dataclass(frozen=True)
class SpeedCalibration:
    min_margin: str
    noise_multiplier: str
    max_noise: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "min_margin", _decimal(self.min_margin, field="min_margin", positive=True)
        )
        object.__setattr__(
            self,
            "noise_multiplier",
            _decimal(self.noise_multiplier, field="noise_multiplier", positive=True),
        )
        object.__setattr__(self, "max_noise", _decimal(self.max_noise, field="max_noise"))
        if decimal_value(self.min_margin) >= 1 or decimal_value(self.max_noise) >= 1:
            raise CalibrationError("speed margin/noise thresholds must be below one")

    def to_dict(self) -> dict[str, str]:
        return {
            "max_noise": self.max_noise,
            "min_margin": self.min_margin,
            "noise_multiplier": self.noise_multiplier,
        }

    @classmethod
    def from_dict(cls, value: object) -> "SpeedCalibration":
        raw = _strict(
            value,
            frozenset({"min_margin", "noise_multiplier", "max_noise"}),
            label="speed calibration",
        )
        return cls(
            min_margin=raw["min_margin"],
            noise_multiplier=raw["noise_multiplier"],
            max_noise=raw["max_noise"],
        )


@dataclass(frozen=True)
class MetricCalibration:
    name: str
    direction: str
    stock_envelope: str
    candidate_delta: str
    absolute_floor: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, field="metric name"))
        if self.name not in _METRIC_DIRECTIONS or self.direction != _METRIC_DIRECTIONS[self.name]:
            raise CalibrationError("metric name/direction is unsupported")
        object.__setattr__(
            self,
            "stock_envelope",
            _decimal(self.stock_envelope, field=f"{self.name} stock_envelope"),
        )
        object.__setattr__(
            self,
            "candidate_delta",
            _decimal(self.candidate_delta, field=f"{self.name} candidate_delta"),
        )
        if self.absolute_floor is not None:
            object.__setattr__(
                self,
                "absolute_floor",
                _decimal(self.absolute_floor, field=f"{self.name} absolute_floor"),
            )
        if self.direction == "lower" and self.absolute_floor is not None:
            raise CalibrationError("a lower-is-better metric cannot declare an absolute floor")
        values = (decimal_value(self.stock_envelope), decimal_value(self.candidate_delta))
        if self.name in _UNIT_METRICS and any(value > 1 for value in values):
            raise CalibrationError("rate metric thresholds must not exceed one")
        if self.absolute_floor is not None and decimal_value(self.absolute_floor) > 1:
            raise CalibrationError("absolute quality floor must not exceed one")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "absolute_floor": self.absolute_floor,
            "candidate_delta": self.candidate_delta,
            "direction": self.direction,
            "name": self.name,
            "stock_envelope": self.stock_envelope,
        }

    @classmethod
    def from_dict(cls, value: object) -> "MetricCalibration":
        raw = _strict(
            value,
            frozenset(
                {"name", "direction", "stock_envelope", "candidate_delta", "absolute_floor"}
            ),
            label="metric calibration",
        )
        return cls(
            name=raw["name"],
            direction=raw["direction"],
            stock_envelope=raw["stock_envelope"],
            candidate_delta=raw["candidate_delta"],
            absolute_floor=raw["absolute_floor"],
        )


@dataclass(frozen=True)
class CalibrationControl:
    kind: str
    seed_digest: str
    raw_evidence_digest: str
    expected_outcome: str

    def __post_init__(self) -> None:
        if self.kind not in _CONTROL_OUTCOMES:
            raise CalibrationError("control kind must be stock, positive, or negative")
        if self.expected_outcome != _CONTROL_OUTCOMES[self.kind]:
            raise CalibrationError("control outcome disagrees with its registered kind")
        object.__setattr__(self, "seed_digest", _digest(self.seed_digest, field="seed_digest"))
        object.__setattr__(
            self,
            "raw_evidence_digest",
            _digest(self.raw_evidence_digest, field="raw_evidence_digest"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "expected_outcome": self.expected_outcome,
            "kind": self.kind,
            "raw_evidence_digest": self.raw_evidence_digest,
            "seed_digest": self.seed_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CalibrationControl":
        raw = _strict(
            value,
            frozenset({"kind", "seed_digest", "raw_evidence_digest", "expected_outcome"}),
            label="calibration control",
        )
        return cls(**raw)


@dataclass(frozen=True)
class CalibrationManifest:
    context: CalibrationContext
    algorithm_id: str
    status: str
    speed: SpeedCalibration
    quality_metrics: tuple[MetricCalibration, ...]
    familywise_z: str
    raw_evidence_digest: str
    seed_digests: tuple[str, ...]
    controls: tuple[CalibrationControl, ...]
    policy_version: str = CALIBRATION_POLICY_VERSION
    schema_version: int = CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.context) is not CalibrationContext or type(self.speed) is not SpeedCalibration:
            raise CalibrationError("calibration context/speed policy is not typed")
        if (
            self.policy_version != CALIBRATION_POLICY_VERSION
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise CalibrationError("unsupported calibration policy/schema version")
        object.__setattr__(self, "algorithm_id", _name(self.algorithm_id, field="algorithm_id"))
        if self.status not in _STATUSES:
            raise CalibrationError("calibration status must be frozen or provisional")
        object.__setattr__(
            self,
            "familywise_z",
            _decimal(self.familywise_z, field="familywise_z", positive=True),
        )
        if decimal_value(self.familywise_z) > 10:
            raise CalibrationError("familywise_z exceeds the supported statistical bound")
        object.__setattr__(
            self,
            "raw_evidence_digest",
            _digest(self.raw_evidence_digest, field="raw_evidence_digest"),
        )
        metrics = tuple(self.quality_metrics)
        seeds = tuple(_digest(value, field="seed digest") for value in self.seed_digests)
        controls = tuple(self.controls)
        if (
            not metrics
            or any(type(row) is not MetricCalibration for row in metrics)
            or tuple(row.name for row in metrics) != tuple(sorted(row.name for row in metrics))
            or len({row.name for row in metrics}) != len(metrics)
        ):
            raise CalibrationError("quality metrics must be unique and name-sorted")
        if not seeds or seeds != tuple(sorted(set(seeds))):
            raise CalibrationError("seed digests must be nonempty, unique, and sorted")
        if (
            len(controls) != 3
            or any(type(row) is not CalibrationControl for row in controls)
            or tuple(row.kind for row in controls) != ("negative", "positive", "stock")
        ):
            raise CalibrationError("controls must be exactly negative, positive, stock")
        object.__setattr__(self, "quality_metrics", metrics)
        object.__setattr__(self, "seed_digests", seeds)
        object.__setattr__(self, "controls", controls)

    @property
    def thresholds_frozen(self) -> bool:
        return self.status == "frozen"

    def require_context(self, expected: CalibrationContext) -> "CalibrationManifest":
        if type(expected) is not CalibrationContext or self.context != expected:
            raise CalibrationError("calibration context is stale or mismatched")
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm_id": self.algorithm_id,
            "context": self.context.to_dict(),
            "controls": [row.to_dict() for row in self.controls],
            "familywise_z": self.familywise_z,
            "policy_version": self.policy_version,
            "quality_metrics": [row.to_dict() for row in self.quality_metrics],
            "raw_evidence_digest": self.raw_evidence_digest,
            "schema_version": self.schema_version,
            "seed_digests": list(self.seed_digests),
            "speed": self.speed.to_dict(),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CalibrationManifest":
        fields = frozenset(
            {
                "algorithm_id", "context", "controls", "familywise_z", "policy_version",
                "quality_metrics", "raw_evidence_digest", "schema_version", "seed_digests",
                "speed", "status",
            }
        )
        raw = _strict(value, fields, label="calibration manifest")
        if not isinstance(raw["quality_metrics"], Sequence) or isinstance(
            raw["quality_metrics"], (str, bytes)
        ):
            raise CalibrationError("quality_metrics must be an array")
        if not isinstance(raw["controls"], Sequence) or isinstance(raw["controls"], (str, bytes)):
            raise CalibrationError("controls must be an array")
        if not isinstance(raw["seed_digests"], Sequence) or isinstance(
            raw["seed_digests"], (str, bytes)
        ):
            raise CalibrationError("seed_digests must be an array")
        return cls(
            context=CalibrationContext.from_dict(raw["context"]),
            algorithm_id=raw["algorithm_id"],
            status=raw["status"],
            speed=SpeedCalibration.from_dict(raw["speed"]),
            quality_metrics=tuple(MetricCalibration.from_dict(row) for row in raw["quality_metrics"]),
            familywise_z=raw["familywise_z"],
            raw_evidence_digest=raw["raw_evidence_digest"],
            seed_digests=tuple(raw["seed_digests"]),
            controls=tuple(CalibrationControl.from_dict(row) for row in raw["controls"]),
            policy_version=raw["policy_version"],
            schema_version=raw["schema_version"],
        )

    @property
    def digest(self) -> str:
        return canonical_digest("optima.qualification.calibration", self.to_dict())


__all__ = [
    "CALIBRATION_POLICY_VERSION",
    "CalibrationContext",
    "CalibrationControl",
    "CalibrationError",
    "CalibrationManifest",
    "MetricCalibration",
    "SpeedCalibration",
    "decimal_value",
]
