"""Content-addressed calibration authority for qualification gates.

Calibration is policy, not a convenient collection of local floats.  A frozen
manifest binds an independently configured validator threshold policy, pristine
reference, exact arena/runtime/hardware/workload context, verifier policy, raw
evidence, seeds, and controls.  Qualification may use its thresholds only when
every binding reopens exactly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path

from optima.eval.evidence_store import (
    DEFAULT_MAX_EVIDENCE_BYTES,
    EvidenceArtifactRef,
    EvidenceStoreError,
    publish_evidence,
    reopen_evidence,
)

from optima.stack_identity import (
    StackIdentityError,
    canonical_digest,
    canonical_json_bytes,
)
from optima._strict import require_digest, require_exact_fields


CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_POLICY_VERSION = "qualification-calibration.v1"
CALIBRATION_EVIDENCE_DOMAIN = "qualification-calibration"
CALIBRATION_EVIDENCE_SCHEMA = "calibration-evidence-set-v1"
CALIBRATION_EVIDENCE_POLICY_VERSION = "qualification-calibration-evidence.v1"
CALIBRATION_THRESHOLD_POLICY_VERSION = "qualification-calibration-thresholds.v1"
_NAME = re.compile(r"[a-z][a-z0-9._-]*\Z")
_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?\Z")
_STATUSES = frozenset({"frozen", "provisional"})
_CONTROL_OUTCOMES = {"stock": "PASS", "positive": "PASS", "negative": "FAIL"}
_CALIBRATION_ALGORITHMS = frozenset({"teacher-familywise-v1"})
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
_CONTROL_ORDER = ("negative", "positive", "stock")


class CalibrationError(ValueError):
    """A calibration manifest is malformed, stale, or not crown-authoritative."""


def _digest(value: object, *, field: str) -> str:
    return require_digest(value, field=field, error=CalibrationError)


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
    return require_exact_fields(value, fields=fields, label=label, error=CalibrationError)


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


@dataclass(frozen=True)
class CalibrationThresholdPolicy:
    """Independently committed validator policy for calibration thresholds.

    These values are economic/referee policy, not facts inferred from an opaque
    evidence artifact.  Raw controls demonstrate and bind their calibration;
    they do not get to redefine the thresholds during reopen.
    """

    context: CalibrationContext
    algorithm_id: str
    status: str
    speed: SpeedCalibration
    quality_metrics: tuple[MetricCalibration, ...]
    familywise_z: str
    policy_version: str = CALIBRATION_THRESHOLD_POLICY_VERSION
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.context) is not CalibrationContext or type(self.speed) is not SpeedCalibration:
            raise CalibrationError("threshold policy context/speed is not typed")
        if (
            self.policy_version != CALIBRATION_THRESHOLD_POLICY_VERSION
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise CalibrationError("unsupported calibration threshold policy/schema version")
        object.__setattr__(self, "algorithm_id", _name(self.algorithm_id, field="algorithm_id"))
        if self.algorithm_id not in _CALIBRATION_ALGORITHMS:
            raise CalibrationError("calibration threshold policy names an unknown algorithm")
        if not isinstance(self.status, str) or self.status not in _STATUSES:
            raise CalibrationError("calibration status must be frozen or provisional")
        object.__setattr__(
            self,
            "familywise_z",
            _decimal(self.familywise_z, field="familywise_z", positive=True),
        )
        if decimal_value(self.familywise_z) > 10:
            raise CalibrationError("familywise_z exceeds the supported statistical bound")
        if not isinstance(self.quality_metrics, Sequence) or isinstance(
            self.quality_metrics, (str, bytes)
        ):
            raise CalibrationError("threshold policy quality metrics must be an array")
        metrics = tuple(self.quality_metrics)
        if not metrics or any(type(row) is not MetricCalibration for row in metrics):
            raise CalibrationError("quality metrics must be unique and name-sorted")
        names = tuple(row.name for row in metrics)
        if names != tuple(sorted(set(names))):
            raise CalibrationError("quality metrics must be unique and name-sorted")
        object.__setattr__(self, "quality_metrics", metrics)

    @classmethod
    def from_manifest(cls, manifest: CalibrationManifest) -> "CalibrationThresholdPolicy":
        if type(manifest) is not CalibrationManifest:
            raise CalibrationError("configured calibration manifest is not exact and typed")
        return cls(
            context=manifest.context,
            algorithm_id=manifest.algorithm_id,
            status=manifest.status,
            speed=manifest.speed,
            quality_metrics=manifest.quality_metrics,
            familywise_z=manifest.familywise_z,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm_id": self.algorithm_id,
            "context": self.context.to_dict(),
            "familywise_z": self.familywise_z,
            "policy_version": self.policy_version,
            "quality_metrics": [row.to_dict() for row in self.quality_metrics],
            "schema_version": self.schema_version,
            "speed": self.speed.to_dict(),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CalibrationThresholdPolicy":
        raw = _strict(
            value,
            frozenset(
                {
                    "algorithm_id",
                    "context",
                    "familywise_z",
                    "policy_version",
                    "quality_metrics",
                    "schema_version",
                    "speed",
                    "status",
                }
            ),
            label="calibration threshold policy",
        )
        metrics = raw["quality_metrics"]
        if not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes)):
            raise CalibrationError("threshold policy quality metrics must be an array")
        return cls(
            context=CalibrationContext.from_dict(raw["context"]),
            algorithm_id=raw["algorithm_id"],
            status=raw["status"],
            speed=SpeedCalibration.from_dict(raw["speed"]),
            quality_metrics=tuple(MetricCalibration.from_dict(row) for row in metrics),
            familywise_z=raw["familywise_z"],
            policy_version=raw["policy_version"],
            schema_version=raw["schema_version"],
        )

    @property
    def digest(self) -> str:
        return canonical_digest("optima.qualification.calibration-threshold-policy", self.to_dict())


@dataclass(frozen=True)
class CalibrationMeasurement:
    """One named raw measurement series from a calibration control."""

    name: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, field="measurement name"))
        if not isinstance(self.values, Sequence) or isinstance(self.values, (str, bytes)):
            raise CalibrationError("calibration measurement values must be an array")
        values = tuple(
            _decimal(value, field=f"{self.name} measurement") for value in self.values
        )
        if not values:
            raise CalibrationError("calibration measurement values must be nonempty")
        object.__setattr__(self, "values", values)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "values": list(self.values)}

    @classmethod
    def from_dict(cls, value: object) -> "CalibrationMeasurement":
        raw = _strict(value, frozenset({"name", "values"}), label="calibration measurement")
        values = raw["values"]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise CalibrationError("calibration measurement values must be an array")
        return cls(name=raw["name"], values=tuple(values))


@dataclass(frozen=True)
class CalibrationObservation:
    """Canonical raw observations for exactly one registered control."""

    control_kind: str
    seed_digest: str
    measurements: tuple[CalibrationMeasurement, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.control_kind, str) or self.control_kind not in _CONTROL_OUTCOMES:
            raise CalibrationError("calibration observation has an unknown control kind")
        object.__setattr__(
            self, "seed_digest", _digest(self.seed_digest, field="observation seed digest")
        )
        if not isinstance(self.measurements, Sequence) or isinstance(
            self.measurements, (str, bytes)
        ):
            raise CalibrationError("calibration measurements must be an array")
        measurements = tuple(self.measurements)
        if (
            not measurements
            or any(type(row) is not CalibrationMeasurement for row in measurements)
        ):
            raise CalibrationError("calibration measurements must be nonempty, unique, and sorted")
        names = tuple(row.name for row in measurements)
        if names != tuple(sorted(set(names))):
            raise CalibrationError("calibration measurements must be nonempty, unique, and sorted")
        sample_counts = {len(row.values) for row in measurements}
        if len(sample_counts) != 1:
            raise CalibrationError("calibration measurement series must have equal sample counts")
        object.__setattr__(self, "measurements", measurements)

    def to_dict(self) -> dict[str, object]:
        return {
            "control_kind": self.control_kind,
            "measurements": [row.to_dict() for row in self.measurements],
            "seed_digest": self.seed_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CalibrationObservation":
        raw = _strict(
            value,
            frozenset({"control_kind", "measurements", "seed_digest"}),
            label="calibration observation",
        )
        rows = raw["measurements"]
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise CalibrationError("calibration measurements must be an array")
        return cls(
            control_kind=raw["control_kind"],
            seed_digest=raw["seed_digest"],
            measurements=tuple(CalibrationMeasurement.from_dict(row) for row in rows),
        )


def _familywise_bounds(values: tuple[str, ...], z: Decimal) -> tuple[Decimal, Decimal]:
    """Return deterministic lower/upper mean bounds for canonical decimals."""

    with localcontext() as context:
        context.prec = 50
        samples = tuple(Decimal(value) for value in values)
        count = Decimal(len(samples))
        mean = sum(samples, Decimal(0)) / count
        if len(samples) == 1:
            error = Decimal(0)
        else:
            variance = sum((value - mean) ** 2 for value in samples) / Decimal(
                len(samples) - 1
            )
            error = (variance / count).sqrt() * z
        return mean - error, mean + error


def _upper_bounded(values: tuple[str, ...], limit: Decimal, z: Decimal) -> str:
    low, high = _familywise_bounds(values, z)
    if high <= limit:
        return "PASS"
    if low > limit:
        return "FAIL"
    return "NO_DECISION"


def _lower_bounded(values: tuple[str, ...], floor: Decimal, z: Decimal) -> str:
    low, high = _familywise_bounds(values, z)
    if low >= floor:
        return "PASS"
    if high < floor:
        return "FAIL"
    return "NO_DECISION"


def _grade_control(
    threshold_policy: CalibrationThresholdPolicy,
    observation: CalibrationObservation,
) -> str:
    """Apply the closed teacher-familywise-v1 control-grade algorithm."""

    if threshold_policy.algorithm_id != "teacher-familywise-v1":
        raise CalibrationError("calibration control grading algorithm is not registered")
    rows = {row.name: row.values for row in observation.measurements}
    z = decimal_value(threshold_policy.familywise_z)
    decisions = [
        _upper_bounded(rows["speed_noise"], decimal_value(threshold_policy.speed.max_noise), z)
    ]
    for metric in threshold_policy.quality_metrics:
        limit = metric.stock_envelope if observation.control_kind == "stock" else metric.candidate_delta
        decisions.append(_upper_bounded(rows[metric.name], decimal_value(limit), z))
        if metric.absolute_floor is not None:
            decisions.append(
                _lower_bounded(
                    rows[f"{metric.name}.absolute"],
                    decimal_value(metric.absolute_floor),
                    z,
                )
            )
    if "FAIL" in decisions:
        return "FAIL"
    if "NO_DECISION" in decisions:
        return "NO_DECISION"
    return "PASS"


def derive_calibration_manifest(
    threshold_policy: CalibrationThresholdPolicy,
    observations: Sequence[CalibrationObservation],
) -> CalibrationManifest:
    """Purely project committed threshold policy and raw controls into a manifest."""

    if type(threshold_policy) is not CalibrationThresholdPolicy:
        raise CalibrationError("calibration threshold policy is not exact and typed")
    observations = tuple(observations)
    if (
        len(observations) != 3
        or any(type(row) is not CalibrationObservation for row in observations)
        or tuple(row.control_kind for row in observations) != _CONTROL_ORDER
    ):
        raise CalibrationError("raw calibration controls must be exactly negative, positive, stock")
    seeds = tuple(sorted(row.seed_digest for row in observations))
    if len(set(seeds)) != 3:
        raise CalibrationError("raw calibration controls must use distinct seeds")
    names = tuple(row.name for row in observations[0].measurements)
    if any(tuple(item.name for item in row.measurements) != names for row in observations):
        raise CalibrationError("calibration controls measured different metric sets")
    required = {"speed_noise", *(row.name for row in threshold_policy.quality_metrics)}
    required.update(
        f"{row.name}.absolute"
        for row in threshold_policy.quality_metrics
        if row.absolute_floor is not None
    )
    if required != set(names):
        raise CalibrationError("raw controls do not exactly cover the registered metric set")
    for observation in observations:
        if _grade_control(threshold_policy, observation) != _CONTROL_OUTCOMES[
            observation.control_kind
        ]:
            raise CalibrationError(
                f"{observation.control_kind} control outcome contradicts raw measurements"
            )
    raw_rows = [row.to_dict() for row in observations]
    raw_digest = canonical_digest(
        "optima.qualification.calibration-raw-evidence",
        {"policy_version": CALIBRATION_EVIDENCE_POLICY_VERSION, "observations": raw_rows},
    )
    controls = tuple(
        CalibrationControl(
            kind=row.control_kind,
            seed_digest=row.seed_digest,
            raw_evidence_digest=canonical_digest(
                "optima.qualification.calibration-control-evidence", row.to_dict()
            ),
            expected_outcome=_CONTROL_OUTCOMES[row.control_kind],
        )
        for row in observations
    )
    return CalibrationManifest(
        context=threshold_policy.context,
        algorithm_id=threshold_policy.algorithm_id,
        status=threshold_policy.status,
        speed=threshold_policy.speed,
        quality_metrics=threshold_policy.quality_metrics,
        familywise_z=threshold_policy.familywise_z,
        raw_evidence_digest=raw_digest,
        seed_digests=seeds,
        controls=controls,
    )


@dataclass(frozen=True)
class CalibrationEvidenceSet:
    """Raw controls bound to external policy and derived-manifest digests."""

    threshold_policy_digest: str
    configured_manifest_digest: str
    observations: tuple[CalibrationObservation, ...]
    policy_version: str = CALIBRATION_EVIDENCE_POLICY_VERSION
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "threshold_policy_digest",
            _digest(self.threshold_policy_digest, field="threshold policy digest"),
        )
        object.__setattr__(
            self,
            "configured_manifest_digest",
            _digest(self.configured_manifest_digest, field="configured manifest digest"),
        )
        if (
            self.policy_version != CALIBRATION_EVIDENCE_POLICY_VERSION
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise CalibrationError("unsupported calibration evidence policy/schema version")
        observations = tuple(self.observations)
        if (
            len(observations) != 3
            or any(type(row) is not CalibrationObservation for row in observations)
            or tuple(row.control_kind for row in observations) != _CONTROL_ORDER
        ):
            raise CalibrationError("raw calibration controls must be exactly negative, positive, stock")
        object.__setattr__(self, "observations", observations)

    @classmethod
    def create(
        cls,
        threshold_policy: CalibrationThresholdPolicy,
        observations: Sequence[CalibrationObservation],
    ) -> "CalibrationEvidenceSet":
        """Bind an independent threshold policy and raw controls into a manifest."""

        rows = tuple(observations)
        manifest = derive_calibration_manifest(threshold_policy, rows)
        return cls(
            threshold_policy_digest=threshold_policy.digest,
            configured_manifest_digest=manifest.digest,
            observations=rows,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "configured_manifest_digest": self.configured_manifest_digest,
            "observations": [row.to_dict() for row in self.observations],
            "policy_version": self.policy_version,
            "schema_version": self.schema_version,
            "threshold_policy_digest": self.threshold_policy_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CalibrationEvidenceSet":
        raw = _strict(
            value,
            frozenset(
                {
                    "configured_manifest_digest",
                    "observations",
                    "policy_version",
                    "schema_version",
                    "threshold_policy_digest",
                }
            ),
            label="calibration evidence set",
        )
        rows = raw["observations"]
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise CalibrationError("calibration observations must be an array")
        return cls(
            threshold_policy_digest=raw["threshold_policy_digest"],
            configured_manifest_digest=raw["configured_manifest_digest"],
            observations=tuple(CalibrationObservation.from_dict(row) for row in rows),
            policy_version=raw["policy_version"],
            schema_version=raw["schema_version"],
        )


def _parse_calibration_evidence(payload: bytes) -> CalibrationEvidenceSet:
    def reject(_value: str) -> None:
        raise CalibrationError("calibration evidence JSON contains a float or nonfinite number")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise CalibrationError("calibration evidence JSON contains a duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_float=reject,
            parse_constant=reject,
            object_pairs_hook=pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"calibration evidence JSON is invalid: {exc}") from None
    try:
        if canonical_json_bytes(value) != payload:
            raise CalibrationError("calibration evidence JSON is not canonically encoded")
    except StackIdentityError as exc:
        raise CalibrationError(f"calibration evidence JSON is invalid: {exc}") from None
    return CalibrationEvidenceSet.from_dict(value)


def publish_calibration_evidence(
    root: str | Path,
    evidence: CalibrationEvidenceSet,
    *,
    max_bytes: int = DEFAULT_MAX_EVIDENCE_BYTES,
) -> EvidenceArtifactRef:
    if type(evidence) is not CalibrationEvidenceSet:
        raise CalibrationError("calibration evidence set is not exact and typed")
    try:
        return publish_evidence(
            root,
            canonical_json_bytes(evidence.to_dict()),
            domain=CALIBRATION_EVIDENCE_DOMAIN,
            media_type="application/json",
            schema=CALIBRATION_EVIDENCE_SCHEMA,
            max_bytes=max_bytes,
        )
    except (EvidenceStoreError, StackIdentityError) as exc:
        raise CalibrationError(f"cannot publish calibration evidence: {exc}") from None


def reopen_calibration_evidence(
    root: str | Path,
    reference: EvidenceArtifactRef,
    *,
    expected_threshold_policy: CalibrationThresholdPolicy,
    expected_manifest: CalibrationManifest,
    expected_context: CalibrationContext,
    max_bytes: int = DEFAULT_MAX_EVIDENCE_BYTES,
) -> CalibrationManifest:
    """Authenticate raw bytes, pure-rederive the manifest, and bind live context."""

    if (
        type(reference) is not EvidenceArtifactRef
        or type(expected_threshold_policy) is not CalibrationThresholdPolicy
        or type(expected_manifest) is not CalibrationManifest
        or type(expected_context) is not CalibrationContext
    ):
        raise CalibrationError(
            "calibration reference, threshold policy, manifest, and context must be exact and typed"
        )
    if (reference.domain, reference.media_type, reference.schema) != (
        CALIBRATION_EVIDENCE_DOMAIN,
        "application/json",
        CALIBRATION_EVIDENCE_SCHEMA,
    ):
        raise CalibrationError("calibration evidence artifact type is not authoritative")
    try:
        payload = reopen_evidence(root, reference, max_bytes=max_bytes)
    except EvidenceStoreError as exc:
        raise CalibrationError(f"cannot reopen calibration evidence: {exc}") from None
    evidence = _parse_calibration_evidence(payload)
    if evidence.threshold_policy_digest != expected_threshold_policy.digest:
        raise CalibrationError("calibration evidence names another validator threshold policy")
    rederived = derive_calibration_manifest(expected_threshold_policy, evidence.observations)
    if evidence.configured_manifest_digest != rederived.digest or rederived != expected_manifest:
        raise CalibrationError("calibration evidence names another configured manifest")
    rederived.require_context(expected_context)
    return rederived


__all__ = [
    "CALIBRATION_EVIDENCE_DOMAIN",
    "CALIBRATION_EVIDENCE_POLICY_VERSION",
    "CALIBRATION_EVIDENCE_SCHEMA",
    "CALIBRATION_POLICY_VERSION",
    "CALIBRATION_THRESHOLD_POLICY_VERSION",
    "CalibrationContext",
    "CalibrationControl",
    "CalibrationEvidenceSet",
    "CalibrationError",
    "CalibrationManifest",
    "CalibrationMeasurement",
    "CalibrationObservation",
    "CalibrationThresholdPolicy",
    "MetricCalibration",
    "SpeedCalibration",
    "decimal_value",
    "derive_calibration_manifest",
    "publish_calibration_evidence",
    "reopen_calibration_evidence",
]
