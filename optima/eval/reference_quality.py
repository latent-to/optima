"""Derive pristine-reference quality only from authenticated per-token evidence.

The public summary objects are projections; ``reopen_reference_quality_evidence``
is the authoritative path, and accepts no candidate-supplied aggregate.
"""

from __future__ import annotations

import json
import math
import statistics
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext
from enum import Enum
from pathlib import Path
from typing import ClassVar

from optima.eval.calibration import (
    CalibrationContext,
    CalibrationManifest,
    MetricCalibration,
    decimal_value,
)
from optima.eval.evidence_store import (
    DEFAULT_MAX_EVIDENCE_BYTES,
    EvidenceArtifactRef,
    EvidenceStoreError,
    reopen_evidence,
)
from optima.stack_identity import (
    StackIdentityError,
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
)

QUALITY_SCHEMA_VERSION = 1
QUALITY_POLICY_VERSION = "pristine-reference-quality.v1"
RAW_QUALITY_DOMAIN = "reference-quality.raw"
RAW_QUALITY_SCHEMA = "optima.reference-quality-raw.v1"
SUPPORT_POLICY_ID = "retained-support-f32-q18.v1"
QUALITY_DECISIONS = frozenset({"PASS", "FAIL", "NO_DECISION"})
_METRICS = frozenset(
    {"mean_nll", "worst_nll", "tail_rate", "topk_kl", "argmax_rate", "coverage_dev", "task_score"}
)
_MAX_PROMPTS = 4096
_MAX_TOKENS = 1_048_576
_MAX_TOPK = 256
_MAX_TASKS = 4096
_MAX_TOKEN_ID = (1 << 63) - 1
_MAX_DECIMAL = Decimal("1000000000000000000")
_MAX_NLL = Decimal("1000000")

class ReferenceQualityError(ValueError):
    """Raw or projected pristine-reference evidence is invalid."""
def _digest(value: object, field: str) -> str:
    try:
        result = require_sha256_hex(value, field=field)
    except StackIdentityError as exc:
        raise ReferenceQualityError(str(exc)) from exc
    if result == "0" * 64:
        raise ReferenceQualityError(f"{field} must not be the all-zero digest")
    return result
def _integer(value: object, field: str, minimum: int = 0, maximum: int = _MAX_TOKEN_ID) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ReferenceQualityError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value
def _decimal(value: object, field: str, maximum: Decimal = _MAX_DECIMAL) -> str:
    if not isinstance(value, str) or len(value) > 96:
        raise ReferenceQualityError(f"{field} must be a bounded canonical decimal string")
    try:
        number = decimal_value(value)
    except ValueError as exc:
        raise ReferenceQualityError(f"{field}: {exc}") from exc
    if not number.is_finite() or number < 0 or number > maximum:
        raise ReferenceQualityError(f"{field} is outside its finite bound")
    return value
def _probability(value: object, field: str) -> str:
    text = _decimal(value, field, Decimal(1))
    if decimal_value(text) <= 0:
        raise ReferenceQualityError(f"{field} must be positive")
    return text
def _strict(value: object, expected: frozenset[str], label: str) -> Mapping[str, object]:
    if (
        not isinstance(value, Mapping)
        or not all(isinstance(key, str) for key in value)
        or frozenset(value) != expected
    ):
        raise ReferenceQualityError(f"{label} fields do not match the schema")
    return value
def _array(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReferenceQualityError(f"{label} must be an array")
    return value
def _encode(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if value is None or type(value) in {str, bool, int}:
        return value
    raise ReferenceQualityError(f"canonical evidence contains {type(value).__name__}")
def _load(cls, value: object, label: str, **converters):
    names = frozenset(field.name for field in fields(cls))
    raw = _strict(value, names, label)
    return cls(**{
        name: converters[name](raw[name]) if name in converters else raw[name]
        for name in names
    })
def _rows(value: object, parser, label: str) -> tuple:
    return tuple(parser(row) for row in _array(value, label))
def _ordered(rows: tuple, keys: tuple, label: str, maximum: int) -> None:
    if not 1 <= len(rows) <= maximum or len(set(keys)) != len(keys) or keys != tuple(sorted(keys)):
        raise ReferenceQualityError(f"{label} must be bounded, unique, and canonically ordered")
def _sumd(values) -> Decimal:
    with localcontext() as context:
        context.prec = 96
        return sum(values, Decimal(0))
def _ratio(numerator: Decimal, denominator: int) -> Decimal:
    with localcontext() as context:
        context.prec = 96
        return numerator / denominator
def _text(value: Decimal, field: str, maximum: Decimal = _MAX_DECIMAL) -> str:
    if not value.is_finite() or value < 0 or value > maximum:
        raise ReferenceQualityError(f"computed {field} is nonfinite or out of range")
    with localcontext() as context:
        context.prec = 96
        text = format(value.normalize(), "f")
    if text in {"", "-0"}:
        text = "0"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return _decimal(text, field, maximum)

class _Record:
    _domain: ClassVar[str | None] = None
    def to_dict(self) -> dict[str, object]:
        result = _encode(self)
        assert isinstance(result, dict)
        return result
    @property
    def digest(self) -> str:
        if self._domain is None:
            raise AttributeError("this projection has no standalone digest")
        return canonical_digest(self._domain, self.to_dict())
@dataclass(frozen=True)
class TeacherNLLEvidence(_Record):
    token_count: int
    nll_sum: str
    nll_max: str
    tail_count: int
    def __post_init__(self) -> None:
        object.__setattr__(self, "token_count", _integer(self.token_count, "token_count", 1, _MAX_TOKENS))
        object.__setattr__(self, "nll_sum", _decimal(self.nll_sum, "nll_sum"))
        object.__setattr__(self, "nll_max", _decimal(self.nll_max, "nll_max", _MAX_NLL))
        object.__setattr__(self, "tail_count", _integer(self.tail_count, "tail_count", 0, self.token_count))
        if decimal_value(self.nll_sum) > decimal_value(self.nll_max) * self.token_count:
            raise ReferenceQualityError("NLL sum exceeds token_count * max")
    @property
    def mean(self) -> float:
        return float(_ratio(decimal_value(self.nll_sum), self.token_count))
    @property
    def worst(self) -> float:
        return float(decimal_value(self.nll_max))
    @property
    def tail_rate(self) -> float:
        return self.tail_count / self.token_count
    @classmethod
    def from_dict(cls, value: object) -> "TeacherNLLEvidence":
        return _load(cls, value, "teacher NLL")
@dataclass(frozen=True)
class RolloutKLEvidence(_Record):
    position_count: int
    mean_kl: str
    max_kl: str
    p99_kl: str
    argmax_disagreements: int
    mean_coverage_deviation: str
    def __post_init__(self) -> None:
        object.__setattr__(self, "position_count", _integer(self.position_count, "position_count", 1, _MAX_TOKENS))
        for field in ("mean_kl", "max_kl", "p99_kl", "mean_coverage_deviation"):
            object.__setattr__(self, field, _decimal(getattr(self, field), field))
        object.__setattr__(self, "argmax_disagreements", _integer(self.argmax_disagreements, "argmax disagreements", 0, self.position_count))
        if decimal_value(self.mean_kl) > decimal_value(self.max_kl) or decimal_value(self.p99_kl) > decimal_value(self.max_kl):
            raise ReferenceQualityError("KL aggregates are inconsistent")
        if decimal_value(self.mean_coverage_deviation) > 1:
            raise ReferenceQualityError("coverage deviation exceeds one")
    @property
    def argmax_rate(self) -> float:
        return self.argmax_disagreements / self.position_count
    @classmethod
    def from_dict(cls, value: object) -> "RolloutKLEvidence":
        return _load(cls, value, "rollout KL")
@dataclass(frozen=True)
class HiddenTaskEvidence(_Record):
    score: str
    total: int
    def __post_init__(self) -> None:
        object.__setattr__(self, "score", _decimal(self.score, "hidden score"))
        object.__setattr__(self, "total", _integer(self.total, "hidden total", 0, _MAX_TASKS))
        if decimal_value(self.score) > self.total:
            raise ReferenceQualityError("hidden score exceeds total")
    @property
    def rate(self) -> float | None:
        return float(_ratio(decimal_value(self.score), self.total)) if self.total else None
    @classmethod
    def from_dict(cls, value: object) -> "HiddenTaskEvidence":
        return _load(cls, value, "hidden task")
@dataclass(frozen=True)
class RolloutQualityEvidence(_Record):
    teacher_nll: TeacherNLLEvidence
    rollout_kl: RolloutKLEvidence
    hidden_task: HiddenTaskEvidence
    def __post_init__(self) -> None:
        if not all(type(row) is expected for row, expected in (
            (self.teacher_nll, TeacherNLLEvidence),
            (self.rollout_kl, RolloutKLEvidence),
            (self.hidden_task, HiddenTaskEvidence),
        )) or self.teacher_nll.token_count != self.rollout_kl.position_count:
            raise ReferenceQualityError("rollout quality types or coverage differ")
    @classmethod
    def from_dict(cls, value: object) -> "RolloutQualityEvidence":
        return _load(cls, value, "rollout quality", teacher_nll=TeacherNLLEvidence.from_dict,
                     rollout_kl=RolloutKLEvidence.from_dict, hidden_task=HiddenTaskEvidence.from_dict)
@dataclass(frozen=True)
class PromptQualityEvidence(_Record):
    prompt_digest: str
    baseline: RolloutQualityEvidence
    candidate: RolloutQualityEvidence
    stock_control: RolloutQualityEvidence
    exact_token_matches: int
    exact_token_total: int
    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_digest", _digest(self.prompt_digest, "prompt_digest"))
        rows = (self.baseline, self.candidate, self.stock_control)
        if not all(type(row) is RolloutQualityEvidence for row in rows) or len({row.teacher_nll.token_count for row in rows}) != 1:
            raise ReferenceQualityError("prompt rollout types or coverage differ")
        total = next(iter({row.teacher_nll.token_count for row in rows}))
        object.__setattr__(self, "exact_token_total", _integer(self.exact_token_total, "exact total", 1, total))
        object.__setattr__(self, "exact_token_matches", _integer(self.exact_token_matches, "exact matches", 0, total))
        if self.exact_token_total != total:
            raise ReferenceQualityError("exact-token coverage differs from rollout")
    @classmethod
    def from_dict(cls, value: object) -> "PromptQualityEvidence":
        return _load(cls, value, "prompt quality", baseline=RolloutQualityEvidence.from_dict,
                     candidate=RolloutQualityEvidence.from_dict, stock_control=RolloutQualityEvidence.from_dict)
@dataclass(frozen=True)
class ReferenceQualityEvidence(_Record):
    _domain: ClassVar[str] = "optima.qualification.reference-quality"
    reference_manifest_digest: str
    calibration_digest: str
    raw_evidence_digest: str
    prompts: tuple[PromptQualityEvidence, ...]
    hidden_tasks_present: bool
    policy_version: str = QUALITY_POLICY_VERSION
    schema_version: int = QUALITY_SCHEMA_VERSION
    def __post_init__(self) -> None:
        for field in ("reference_manifest_digest", "calibration_digest", "raw_evidence_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if self.policy_version != QUALITY_POLICY_VERSION or type(self.schema_version) is not int or self.schema_version != 1:
            raise ReferenceQualityError("unsupported reference quality policy/schema")
        if type(self.hidden_tasks_present) is not bool:
            raise ReferenceQualityError("hidden_tasks_present must be boolean")
        prompts = tuple(self.prompts)
        if not all(type(row) is PromptQualityEvidence for row in prompts):
            raise ReferenceQualityError("prompt evidence is not typed")
        _ordered(prompts, tuple(row.prompt_digest for row in prompts), "prompts", _MAX_PROMPTS)
        for prompt in prompts:
            totals = tuple(row.hidden_task.total for row in (prompt.baseline, prompt.candidate, prompt.stock_control))
            if (self.hidden_tasks_present and not all(totals)) or (not self.hidden_tasks_present and any(totals)):
                raise ReferenceQualityError("hidden-task declaration/coverage mismatch")
        object.__setattr__(self, "prompts", prompts)
    @classmethod
    def from_dict(cls, value: object) -> "ReferenceQualityEvidence":
        return _load(cls, value, "reference quality evidence",
                     prompts=lambda rows: _rows(rows, PromptQualityEvidence.from_dict, "prompts"))
@dataclass(frozen=True)
class ReferenceQualityRawBinding(_Record):
    _domain: ClassVar[str] = "optima.qualification.reference-quality-raw-binding"
    qualification_identity_digest: str
    reference_manifest_digest: str
    calibration_digest: str
    selection_digest: str
    candidate_lifecycle_digest: str
    selected_trajectory_digest: str
    selected_trajectory_projection_digest: str
    selected_prompt_digests: tuple[str, ...]
    t_session_digest: str
    t_request_sha256: str
    support_policy_digest: str
    hidden_task_plan_digest: str
    nll_tail_threshold: str
    tokens_per_prompt: int
    topk_width: int
    hidden_tasks_per_prompt: int
    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name.endswith("_digest") or field.name == "t_request_sha256":
                object.__setattr__(self, field.name, _digest(getattr(self, field.name), field.name))
        prompts = tuple(_digest(value, "selected prompt") for value in self.selected_prompt_digests)
        _ordered(prompts, prompts, "selected prompts", _MAX_PROMPTS)
        object.__setattr__(self, "selected_prompt_digests", prompts)
        object.__setattr__(self, "nll_tail_threshold", _decimal(self.nll_tail_threshold, "NLL tail threshold", _MAX_NLL))
        object.__setattr__(self, "tokens_per_prompt", _integer(self.tokens_per_prompt, "tokens per prompt", 1, _MAX_TOKENS))
        object.__setattr__(self, "topk_width", _integer(self.topk_width, "top-k width", 1, _MAX_TOPK))
        object.__setattr__(self, "hidden_tasks_per_prompt", _integer(self.hidden_tasks_per_prompt, "hidden tasks per prompt", 0, _MAX_TASKS))
    @classmethod
    def from_dict(cls, value: object) -> "ReferenceQualityRawBinding":
        return _load(cls, value, "raw quality binding",
                     selected_prompt_digests=lambda rows: tuple(_array(rows, "selected prompts")))
@dataclass(frozen=True)
class TokenProbability(_Record):
    token_id: int
    probability: str
    def __post_init__(self) -> None:
        object.__setattr__(self, "token_id", _integer(self.token_id, "token_id"))
        object.__setattr__(self, "probability", _probability(self.probability, "probability"))
    @classmethod
    def from_dict(cls, value: object) -> "TokenProbability":
        return _load(cls, value, "token probability")
@dataclass(frozen=True)
class NormalizedTopKDistribution(_Record):
    entries: tuple[TokenProbability, ...]
    tail_probability: str
    true_argmax_token_id: int
    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if not all(type(row) is TokenProbability for row in entries):
            raise ReferenceQualityError("top-k entries are not typed")
        _ordered(entries, tuple(row.token_id for row in entries), "top-k entries", _MAX_TOPK)
        object.__setattr__(self, "tail_probability", _probability(self.tail_probability, "tail probability"))
        total = _sumd([decimal_value(row.probability) for row in entries] + [decimal_value(self.tail_probability)])
        if total != 1:
            raise ReferenceQualityError("top-k entries and tail must normalize exactly to one")
        object.__setattr__(self, "true_argmax_token_id", _integer(self.true_argmax_token_id, "true argmax token"))
        object.__setattr__(self, "entries", entries)
    @property
    def support(self) -> tuple[int, ...]:
        return tuple(row.token_id for row in self.entries)
    @classmethod
    def from_dict(cls, value: object) -> "NormalizedTopKDistribution":
        return _load(cls, value, "normalized top-k", entries=lambda rows: _rows(rows, TokenProbability.from_dict, "top-k entries"))


def retained_support_policy_digest() -> str:
    return canonical_digest(
        "optima.qualification.support-policy",
        {"clamp_min_logprob": "-80", "mass_units": 10**18, "policy_id": SUPPORT_POLICY_ID},
    )


def distribution_from_f32_logprobs(
    support_ids: tuple[int, ...],
    logprobs: tuple[float, ...],
    *,
    true_argmax_token_id: int,
) -> NormalizedTopKDistribution:
    """Project binary32 support logprobs into an exact Q=1e18 distribution."""

    support = tuple(_integer(value, "support token") for value in support_ids)
    values = tuple(logprobs)
    if (
        not support
        or support != tuple(sorted(set(support)))
        or len(values) != len(support)
        or any(type(value) is not float or not math.isfinite(value) or value > 1e-4 for value in values)
    ):
        raise ReferenceQualityError("retained support/logprobs are malformed")
    values = tuple(struct.unpack(">f", struct.pack(">f", value))[0] for value in values)
    quantum = 10**18
    with localcontext() as context:
        context.prec = 96
        context.rounding = ROUND_HALF_EVEN
        weights = [max(1, int((max(Decimal.from_float(value), Decimal(-80)).exp() * quantum)
                              .to_integral_value(rounding=ROUND_HALF_EVEN))) for value in values]
        total = sum(weights)
        if total >= quantum:
            target = quantum - 1
            exact = [Decimal(weight) * target / total for weight in weights]
            units = [int(value.to_integral_value(rounding=ROUND_FLOOR)) for value in exact]
            remaining = target - sum(units)
            order = sorted(
                range(len(units)),
                key=lambda index: (-(exact[index] - units[index]), support[index]),
            )
            for index in order[:remaining]:
                units[index] += 1
            weights, tail = units, 1
        else:
            tail = quantum - total
    def probability(units: int) -> str:
        return _text(Decimal(units) / quantum, "support probability", Decimal(1))
    return NormalizedTopKDistribution(
        tuple(TokenProbability(token, probability(units)) for token, units in zip(support, weights)),
        probability(tail),
        true_argmax_token_id,
    )


def target_nll_from_f32(logprob: float) -> str:
    if type(logprob) is not float or not math.isfinite(logprob) or logprob > 1e-4:
        raise ReferenceQualityError("target logprob is malformed")
    return _text(max(Decimal(0), -Decimal.from_float(logprob)), "target NLL", _MAX_NLL)
@dataclass(frozen=True)
class RawTokenEvidence(_Record):
    position: int
    token_id: int
    target_nll: str
    teacher_topk: NormalizedTopKDistribution
    rollout_topk: NormalizedTopKDistribution
    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _integer(self.position, "position", 0, _MAX_TOKENS - 1))
        object.__setattr__(self, "token_id", _integer(self.token_id, "token_id"))
        object.__setattr__(self, "target_nll", _decimal(self.target_nll, "target NLL", _MAX_NLL))
        if type(self.teacher_topk) is not NormalizedTopKDistribution or type(self.rollout_topk) is not NormalizedTopKDistribution:
            raise ReferenceQualityError("raw token distributions are not typed")
        if self.teacher_topk.support != self.rollout_topk.support:
            raise ReferenceQualityError("teacher and rollout top-k support differs")
    @classmethod
    def from_dict(cls, value: object) -> "RawTokenEvidence":
        return _load(cls, value, "raw token", teacher_topk=NormalizedTopKDistribution.from_dict,
                     rollout_topk=NormalizedTopKDistribution.from_dict)
@dataclass(frozen=True)
class RawHiddenTaskResult(_Record):
    task_digest: str
    passed: bool
    def __post_init__(self) -> None:
        object.__setattr__(self, "task_digest", _digest(self.task_digest, "task_digest"))
        if type(self.passed) is not bool:
            raise ReferenceQualityError("hidden task result must be boolean")
    @classmethod
    def from_dict(cls, value: object) -> "RawHiddenTaskResult":
        return _load(cls, value, "hidden task result")
@dataclass(frozen=True)
class RawRolloutEvidence(_Record):
    tokens: tuple[RawTokenEvidence, ...]
    hidden_tasks: tuple[RawHiddenTaskResult, ...]
    def __post_init__(self) -> None:
        tokens, tasks = tuple(self.tokens), tuple(self.hidden_tasks)
        if not all(type(row) is RawTokenEvidence for row in tokens):
            raise ReferenceQualityError("raw rollout tokens are not typed")
        if not 1 <= len(tokens) <= _MAX_TOKENS or tuple(row.position for row in tokens) != tuple(range(len(tokens))):
            raise ReferenceQualityError("raw rollout token positions are incomplete or reordered")
        if not all(type(row) is RawHiddenTaskResult for row in tasks):
            raise ReferenceQualityError("raw hidden tasks are not typed")
        if tasks:
            _ordered(tasks, tuple(row.task_digest for row in tasks), "hidden tasks", _MAX_TASKS)
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "hidden_tasks", tasks)
    @classmethod
    def from_dict(cls, value: object) -> "RawRolloutEvidence":
        return _load(cls, value, "raw rollout",
                     tokens=lambda rows: _rows(rows, RawTokenEvidence.from_dict, "raw tokens"),
                     hidden_tasks=lambda rows: _rows(rows, RawHiddenTaskResult.from_dict, "raw hidden tasks"))
@dataclass(frozen=True)
class RawPromptQualityEvidence(_Record):
    prompt_digest: str
    baseline: RawRolloutEvidence
    candidate: RawRolloutEvidence
    stock_control: RawRolloutEvidence
    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_digest", _digest(self.prompt_digest, "prompt_digest"))
        rows = (self.baseline, self.candidate, self.stock_control)
        if not all(type(row) is RawRolloutEvidence for row in rows) or len({len(row.tokens) for row in rows}) != 1:
            raise ReferenceQualityError("raw prompt rollout types or token coverage differ")
        if len({tuple(task.task_digest for task in row.hidden_tasks) for row in rows}) != 1:
            raise ReferenceQualityError("raw prompt hidden-task identities differ")
    @classmethod
    def from_dict(cls, value: object) -> "RawPromptQualityEvidence":
        return _load(cls, value, "raw prompt", baseline=RawRolloutEvidence.from_dict,
                     candidate=RawRolloutEvidence.from_dict, stock_control=RawRolloutEvidence.from_dict)
def hidden_task_plan_digest(prompts: tuple[RawPromptQualityEvidence, ...]) -> str:
    return canonical_digest(
        "optima.qualification.hidden-task-plan",
        [{"prompt": row.prompt_digest,
          "tasks": [task.task_digest for task in row.baseline.hidden_tasks]}
         for row in prompts],
    )
def raw_trajectory_projection_digest(prompts: tuple[RawPromptQualityEvidence, ...]) -> str:
    def rollout(row: RawRolloutEvidence) -> dict[str, object]:
        return {
            "output_ids": [token.token_id for token in row.tokens],
            "rollout_topk": [token.rollout_topk.to_dict() for token in row.tokens],
        }
    return canonical_digest(
        "optima.qualification.selected-trajectory-projection",
        {
            "support_policy_digest": retained_support_policy_digest(),
            "prompts": [
                {
                    "prompt": prompt.prompt_digest,
                    "rollouts": [
                        rollout(row)
                        for row in (
                            prompt.baseline,
                            prompt.candidate,
                            prompt.stock_control,
                        )
                    ],
                }
                for prompt in prompts
            ],
        },
    )
@dataclass(frozen=True)
class ReferenceQualityRawArtifact(_Record):
    _domain: ClassVar[str] = "optima.qualification.reference-quality-raw"
    binding: ReferenceQualityRawBinding
    prompts: tuple[RawPromptQualityEvidence, ...]
    policy_version: str = QUALITY_POLICY_VERSION
    schema_version: int = QUALITY_SCHEMA_VERSION
    def __post_init__(self) -> None:
        if type(self.binding) is not ReferenceQualityRawBinding:
            raise ReferenceQualityError("raw quality binding is not typed")
        if self.policy_version != QUALITY_POLICY_VERSION or type(self.schema_version) is not int or self.schema_version != 1:
            raise ReferenceQualityError("unsupported raw quality policy/schema")
        prompts = tuple(self.prompts)
        if not all(type(row) is RawPromptQualityEvidence for row in prompts):
            raise ReferenceQualityError("raw prompts are not typed")
        _ordered(prompts, tuple(row.prompt_digest for row in prompts), "raw prompts", _MAX_PROMPTS)
        if tuple(row.prompt_digest for row in prompts) != self.binding.selected_prompt_digests:
            raise ReferenceQualityError("raw prompts differ from the bound selection")
        hidden = {bool(row.baseline.hidden_tasks) for row in prompts}
        if len(hidden) != 1:
            raise ReferenceQualityError("hidden-task coverage must be all-or-none across prompts")
        for prompt in prompts:
            for rollout in (prompt.baseline, prompt.candidate, prompt.stock_control):
                if (
                    len(rollout.tokens) != self.binding.tokens_per_prompt
                    or len(rollout.hidden_tasks) != self.binding.hidden_tasks_per_prompt
                    or any(
                        len(token.teacher_topk.entries) != self.binding.topk_width
                        or len(token.rollout_topk.entries) != self.binding.topk_width
                        for token in rollout.tokens
                    )
                ):
                    raise ReferenceQualityError("raw quality coverage differs from its binding")
        if hidden_task_plan_digest(prompts) != self.binding.hidden_task_plan_digest:
            raise ReferenceQualityError("raw hidden-task plan differs from its binding")
        if raw_trajectory_projection_digest(prompts) != self.binding.selected_trajectory_projection_digest:
            raise ReferenceQualityError("raw trajectories differ from retained B/C/B-prime evidence")
        object.__setattr__(self, "prompts", prompts)
    @classmethod
    def from_dict(cls, value: object) -> "ReferenceQualityRawArtifact":
        return _load(cls, value, "raw quality artifact", binding=ReferenceQualityRawBinding.from_dict,
                     prompts=lambda rows: _rows(rows, RawPromptQualityEvidence.from_dict, "raw prompts"))
def _distribution_metrics(token: RawTokenEvidence) -> tuple[Decimal, Decimal, bool]:
    with localcontext() as context:
        context.prec = 48
        teacher = [decimal_value(row.probability) for row in token.teacher_topk.entries]
        rollout = [decimal_value(row.probability) for row in token.rollout_topk.entries]
        teacher.append(decimal_value(token.teacher_topk.tail_probability))
        rollout.append(decimal_value(token.rollout_topk.tail_probability))
        kl = sum((p * (p / q).ln() for p, q in zip(teacher, rollout)), Decimal(0))
        if kl < 0 and kl >= Decimal("-1e-40"):
            kl = Decimal(0)
        coverage = abs(teacher[-1] - rollout[-1])
    if not kl.is_finite() or not coverage.is_finite() or kl < 0:
        raise ReferenceQualityError("computed token KL/coverage is nonfinite")
    return kl, coverage, token.teacher_topk.true_argmax_token_id != token.rollout_topk.true_argmax_token_id
def _rollout(raw: RawRolloutEvidence, tail_threshold: Decimal) -> RolloutQualityEvidence:
    nlls = [decimal_value(token.target_nll) for token in raw.tokens]
    metrics = [_distribution_metrics(token) for token in raw.tokens]
    kls, coverages = [row[0] for row in metrics], [row[1] for row in metrics]
    ordered_kl = sorted(kls)
    count = len(raw.tokens)
    return RolloutQualityEvidence(
        TeacherNLLEvidence(count, _text(_sumd(nlls), "NLL sum"),
                           _text(max(nlls), "NLL max", _MAX_NLL), sum(value > tail_threshold for value in nlls)),
        RolloutKLEvidence(count, _text(_ratio(_sumd(kls), count), "mean KL"),
                          _text(max(kls), "max KL"),
                          _text(ordered_kl[min(count - 1, int(Decimal("0.99") * count))], "p99 KL"),
                          sum(row[2] for row in metrics),
                          _text(_ratio(_sumd(coverages), count), "coverage deviation", Decimal(1))),
        HiddenTaskEvidence(str(sum(task.passed for task in raw.hidden_tasks)), len(raw.hidden_tasks)),
    )
def _derive(raw: ReferenceQualityRawArtifact, raw_digest: str) -> ReferenceQualityEvidence:
    threshold = decimal_value(raw.binding.nll_tail_threshold)
    prompts = []
    for prompt in raw.prompts:
        baseline = _rollout(prompt.baseline, threshold)
        candidate = _rollout(prompt.candidate, threshold)
        control = _rollout(prompt.stock_control, threshold)
        prompts.append(PromptQualityEvidence(
            prompt.prompt_digest, baseline, candidate, control,
            sum(left.token_id == right.token_id for left, right in zip(prompt.baseline.tokens, prompt.candidate.tokens)),
            len(prompt.baseline.tokens),
        ))
    return ReferenceQualityEvidence(raw.binding.reference_manifest_digest,
                                    raw.binding.calibration_digest, _digest(raw_digest, "raw evidence digest"),
                                    tuple(prompts), bool(raw.prompts[0].baseline.hidden_tasks))
def _parse_canonical(payload: bytes) -> ReferenceQualityRawArtifact:
    def reject(_value: str):
        raise ReferenceQualityError("raw quality JSON contains a float or nonfinite number")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ReferenceQualityError("raw quality JSON contains a duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), parse_float=reject,
                           parse_constant=reject, object_pairs_hook=pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReferenceQualityError(f"raw quality JSON is invalid: {exc}") from None
    try:
        if canonical_json_bytes(value) != payload:
            raise ReferenceQualityError("raw quality JSON is not canonically encoded")
    except StackIdentityError as exc:
        raise ReferenceQualityError(f"raw quality JSON is invalid: {exc}") from None
    return ReferenceQualityRawArtifact.from_dict(value)

def reopen_reference_quality_evidence(
    root: str | Path,
    reference: EvidenceArtifactRef,
    *,
    expected_binding: ReferenceQualityRawBinding,
    max_bytes: int = DEFAULT_MAX_EVIDENCE_BYTES,
) -> ReferenceQualityEvidence:
    """Authenticate canonical raw bytes, bind their authority, and derive summaries."""

    if type(reference) is not EvidenceArtifactRef or type(expected_binding) is not ReferenceQualityRawBinding:
        raise ReferenceQualityError("raw quality reference/binding is not typed")
    if (reference.domain, reference.media_type, reference.schema) != (
        RAW_QUALITY_DOMAIN, "application/json", RAW_QUALITY_SCHEMA
    ):
        raise ReferenceQualityError("raw quality artifact type is not authoritative")
    try:
        payload = reopen_evidence(root, reference, max_bytes=max_bytes)
    except EvidenceStoreError as exc:
        raise ReferenceQualityError(f"cannot reopen raw quality evidence: {exc}") from None
    raw = _parse_canonical(payload)
    if raw.binding != expected_binding:
        raise ReferenceQualityError("raw quality artifact binding was relabeled or is stale")
    return _derive(raw, reference.sha256)
@dataclass(frozen=True)
class ReferenceQualityVerdict:
    decision: str
    failed_metrics: tuple[str, ...]
    overlapping_metrics: tuple[str, ...]
    candidate_mean_teacher_nll: str
    evidence_digest: str
    calibration_digest: str
    def __post_init__(self) -> None:
        if self.decision not in QUALITY_DECISIONS:
            raise ReferenceQualityError("quality decision is invalid")
        for field in ("evidence_digest", "calibration_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(self, "candidate_mean_teacher_nll", _decimal(self.candidate_mean_teacher_nll, "candidate mean NLL"))
        failed, overlap = tuple(self.failed_metrics), tuple(self.overlapping_metrics)
        if any(not isinstance(name, str) or not name for name in failed + overlap):
            raise ReferenceQualityError("quality metric names must be non-empty strings")
        if failed != tuple(sorted(set(failed))) or overlap != tuple(sorted(set(overlap))):
            raise ReferenceQualityError("quality metric sets must be canonical and unique")
        if set(failed) & set(overlap):
            raise ReferenceQualityError("quality metric cannot fail and overlap")
        expected = "FAIL" if failed else ("NO_DECISION" if overlap else "PASS")
        if self.decision != expected:
            raise ReferenceQualityError("quality decision disagrees with its metric sets")
        object.__setattr__(self, "failed_metrics", failed)
        object.__setattr__(self, "overlapping_metrics", overlap)
def _metric(prompt: PromptQualityEvidence, policy: MetricCalibration) -> tuple[float, float, float]:
    def value(row: RolloutQualityEvidence) -> float:
        values = {
            "mean_nll": row.teacher_nll.mean,
            "worst_nll": row.teacher_nll.worst,
            "tail_rate": row.teacher_nll.tail_rate,
            "topk_kl": float(decimal_value(row.rollout_kl.mean_kl)),
            "argmax_rate": row.rollout_kl.argmax_rate,
            "coverage_dev": float(decimal_value(row.rollout_kl.mean_coverage_deviation)),
        }
        if policy.name == "task_score":
            if row.hidden_task.rate is None:
                raise ReferenceQualityError("task_score requires hidden-task evidence")
            return row.hidden_task.rate
        if policy.name not in values:
            raise ReferenceQualityError(f"unsupported calibrated metric {policy.name!r}")
        return values[policy.name]
    return value(prompt.baseline), value(prompt.candidate), value(prompt.stock_control)
def _bounds(values: list[float], z: float) -> tuple[float, float, float]:
    if not values or not math.isfinite(z) or z <= 0 or any(not math.isfinite(value) for value in values):
        raise ReferenceQualityError("quality statistic or z is empty, nonpositive, or nonfinite")
    mean = statistics.fmean(values)
    error = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    result = mean, mean - z * error, mean + z * error
    if any(not math.isfinite(value) for value in result):
        raise ReferenceQualityError("computed quality bound is nonfinite")
    return result
def _mean_nll(prompts: tuple[PromptQualityEvidence, ...]) -> str:
    total = sum(prompt.candidate.teacher_nll.token_count for prompt in prompts)
    nll = _sumd([decimal_value(prompt.candidate.teacher_nll.nll_sum) for prompt in prompts])
    return _text(_ratio(nll, total), "candidate mean NLL")

def score_reference_quality(
    evidence: ReferenceQualityEvidence,
    *,
    calibration: CalibrationManifest,
    expected_context: CalibrationContext,
) -> ReferenceQualityVerdict:
    """Score a derived summary. Crown authority must first use the reopen path above."""

    if type(evidence) is not ReferenceQualityEvidence or type(calibration) is not CalibrationManifest:
        raise ReferenceQualityError("quality evidence/calibration is not typed")
    try:
        calibration.require_context(expected_context)
    except ValueError as exc:
        raise ReferenceQualityError(str(exc)) from exc
    if evidence.reference_manifest_digest != expected_context.reference_manifest_digest:
        raise ReferenceQualityError("quality evidence names another pristine reference")
    if evidence.calibration_digest != calibration.digest:
        raise ReferenceQualityError("quality evidence names another calibration")
    if any(metric.name not in _METRICS for metric in calibration.quality_metrics):
        raise ReferenceQualityError("calibration contains an unsupported quality metric")
    if not calibration.thresholds_frozen:
        return ReferenceQualityVerdict("NO_DECISION", (), ("calibration.provisional",),
                                       _mean_nll(evidence.prompts), evidence.digest, calibration.digest)

    failed, overlap = set(), set()
    z = float(decimal_value(calibration.familywise_z))
    if not math.isfinite(z) or not 0 < z <= 10:
        raise ReferenceQualityError("familywise z is nonfinite or outside (0, 10]")
    for policy in calibration.quality_metrics:
        triplets = [_metric(prompt, policy) for prompt in evidence.prompts]
        stock = [abs(baseline - control) for baseline, _candidate, control in triplets]
        regressions = ([candidate - baseline for baseline, candidate, _control in triplets]
                       if policy.direction == "lower" else
                       [baseline - candidate for baseline, candidate, _control in triplets])
        if _bounds(stock, z)[2] > float(decimal_value(policy.stock_envelope)):
            overlap.add(f"{policy.name}.stock_drift")
            continue
        low, high = _bounds(regressions, z)[1:]
        limit = float(decimal_value(policy.candidate_delta))
        if low > limit:
            failed.add(policy.name)
        elif high > limit:
            overlap.add(policy.name)
        if policy.absolute_floor is not None:
            candidate = [value for _baseline, value, _control in triplets]
            low, high = _bounds(candidate, z)[1:]
            floor = float(decimal_value(policy.absolute_floor))
            if high < floor:
                failed.add(f"{policy.name}.absolute_floor")
            elif low < floor:
                overlap.add(f"{policy.name}.absolute_floor")
    decision = "FAIL" if failed else ("NO_DECISION" if overlap else "PASS")
    return ReferenceQualityVerdict(decision, tuple(sorted(failed)), tuple(sorted(overlap)),
                                   _mean_nll(evidence.prompts), evidence.digest, calibration.digest)

__all__ = [
    "HiddenTaskEvidence", "NormalizedTopKDistribution", "PromptQualityEvidence",
    "QUALITY_DECISIONS", "QUALITY_POLICY_VERSION", "QUALITY_SCHEMA_VERSION",
    "RAW_QUALITY_DOMAIN", "RAW_QUALITY_SCHEMA", "RawHiddenTaskResult",
    "RawPromptQualityEvidence", "RawRolloutEvidence", "RawTokenEvidence",
    "ReferenceQualityError", "ReferenceQualityEvidence", "ReferenceQualityRawArtifact",
    "ReferenceQualityRawBinding", "ReferenceQualityVerdict", "RolloutKLEvidence",
    "RolloutQualityEvidence", "TeacherNLLEvidence", "TokenProbability",
    "SUPPORT_POLICY_ID", "distribution_from_f32_logprobs", "hidden_task_plan_digest",
    "retained_support_policy_digest", "reopen_reference_quality_evidence",
    "score_reference_quality",
    "target_nll_from_f32",
]
