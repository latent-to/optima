"""Bounded, non-executable wire protocol for one isolated engine session.

The host controller and the in-container worker exchange strict JSON control
frames and fixed-width binary token evidence.  The protocol carries no Python
objects, worker timing, verdict, score, hidden quality input, or model-generated
text.  A discovery-only ready frame may carry the bounded
stock-driver scheduler-membership proof required to identify the activated
overlay.  The module remains independent of evaluator and chain packages so
importing it cannot pull candidate or inference-runtime code into the trusted
controller.
"""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from optima.seams import normalize_seam_bindings
from optima.stack_identity import canonical_digest
from optima._strict import require_digest

if TYPE_CHECKING:
    from optima.discovery_overlay import DiscoveryActivationReceipt


SESSION_SCHEMA = "optima-isolated-engine-session-v1"
CONTROL_MAGIC = b"OES1"
EVIDENCE_MAGIC = b"OEE1"
FRAME_HEADER_BYTES = 8

MAX_INIT_BYTES = 64 * 1024
MAX_CONTROL_BYTES = 64 * 1024
MAX_BATCH_REQUEST_BYTES = 128 * 1024 * 1024
MAX_BATCH_RESPONSE_BYTES = 512 * 1024 * 1024
MAX_JSON_NESTING = 24
MAX_JSON_ITEMS = 250_000
MAX_PROMPTS_PER_BATCH = 4096
MAX_PROMPT_CHARS = 2_000_000
MAX_TOTAL_PROMPT_CHARS = 96_000_000
MAX_NEW_TOKENS = 32_768
MAX_TOP_LOGPROBS = 4096
MAX_ERROR_CHARS = 16_384
MAX_AUDIT_RECEIPTS = 4_096

CONTAINER_MODEL_PATH = "/optima/input/model"

_HEX_128 = re.compile(r"[0-9a-f]{32}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9_.:+/@-]{1,256}\Z")
_ARCHITECTURE = re.compile(r"sm[0-9]{2,3}[a-z]?\Z")

# A reviewed extension of this table is required before a new runtime option
# can cross the hostile boundary.  Arbitrary ``sglang.Engine`` kwargs are not a
# protocol feature.
_ENGINE_KWARG_KINDS: Mapping[str, str] = {
    "chunked_prefill_size": "positive_int",
    "context_length": "positive_int",
    "cuda_graph_backend_prefill": "token",
    "disable_radix_cache": "bool",
    "enable_flashinfer_allreduce_fusion": "bool",
    "kv_cache_dtype": "token",
    "max_prefill_tokens": "positive_int",
    "page_size": "positive_int",
    "quantization": "token",
    "trust_remote_code": "bool",
}

ENGINE_CONFIG_FIELDS = frozenset("""
attention_backend deterministic disable_cuda_graph disable_custom_all_reduce dtype
engine_kwargs log_level max_running_requests mem_fraction_static model_path
moe_runner_backend seam_bindings tp_size
""".split())

PREFLIGHT_FACT_FIELDS = frozenset("""
engine_config_digest gpu_architectures launch_digest loopback_only
model_content_digest model_manifest_digest model_revision_digest
private_writable_cache read_only_inputs runtime_digest sglang_version stack_digest
topology_digest tree_digest worker_distribution_digest
""".split())


class SessionProtocolError(ValueError):
    """A session value or frame is malformed, ambiguous, or out of bounds."""


@dataclass(frozen=True)
class SlotAuditControl:
    """Seed-independent controls that both qualification passes must reproduce."""

    sample_rate_ppm: int
    minimum_calls: int
    expected_slots: tuple[str, ...]
    expected_member_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_rate_ppm",
            _bounded_int(
                self.sample_rate_ppm,
                field_name="audit sample_rate_ppm",
                minimum=1,
                maximum=1_000_000,
            ),
        )
        object.__setattr__(
            self,
            "minimum_calls",
            _bounded_int(
                self.minimum_calls,
                field_name="audit minimum_calls",
                minimum=1,
                maximum=1_000_000_000,
            ),
        )
        slots = tuple(self.expected_slots)
        if (
            not slots
            or len(slots) > MAX_AUDIT_RECEIPTS
            or slots != tuple(sorted(set(slots)))
            or any(_TOKEN.fullmatch(slot) is None for slot in slots)
        ):
            raise SessionProtocolError(
                "audit expected_slots must be a nonempty sorted unique token array"
            )
        object.__setattr__(self, "expected_slots", slots)
        object.__setattr__(
            self,
            "expected_member_count",
            _bounded_int(
                self.expected_member_count,
                field_name="audit expected_member_count",
                minimum=1,
                maximum=64,
            ),
        )
        if len(slots) * self.expected_member_count > MAX_AUDIT_RECEIPTS:
            raise SessionProtocolError("audit slot/member receipt bound is exceeded")

    def to_dict(self) -> dict[str, object]:
        return {
            "expected_member_count": self.expected_member_count,
            "expected_slots": list(self.expected_slots),
            "minimum_calls": self.minimum_calls,
            "sample_rate_ppm": self.sample_rate_ppm,
        }

    @classmethod
    def from_dict(cls, value: object) -> "SlotAuditControl":
        row = _exact_object(
            value,
            fields=frozenset(
                "expected_member_count expected_slots minimum_calls sample_rate_ppm".split()
            ),
            label="slot audit control",
        )
        slots = row["expected_slots"]
        if not isinstance(slots, list):
            raise SessionProtocolError("audit expected_slots must be an array")
        return cls(
            row["sample_rate_ppm"],
            row["minimum_calls"],
            tuple(slots),
            row["expected_member_count"],
        )  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("optima.eval.slot-audit-control.v1", self.to_dict())


@dataclass(frozen=True)
class SlotAuditPolicy:
    """Validator-owned policy for one separate untimed candidate audit role."""

    validator_seed: str
    sample_rate_ppm: int
    minimum_calls: int
    expected_slots: tuple[str, ...]
    expected_member_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "validator_seed",
            _binding_id(self.validator_seed, field_name="audit validator_seed"),
        )
        control = SlotAuditControl(
            self.sample_rate_ppm,
            self.minimum_calls,
            self.expected_slots,
            self.expected_member_count,
        )
        for field in SlotAuditControl.__dataclass_fields__:
            object.__setattr__(self, field, getattr(control, field))

    @property
    def control(self) -> SlotAuditControl:
        return SlotAuditControl(
            self.sample_rate_ppm,
            self.minimum_calls,
            self.expected_slots,
            self.expected_member_count,
        )

    def to_dict(self) -> dict[str, object]:
        return {**self.control.to_dict(), "validator_seed": self.validator_seed}

    @classmethod
    def from_dict(cls, value: object) -> "SlotAuditPolicy":
        row = _exact_object(
            value,
            fields=frozenset(
                "expected_member_count expected_slots minimum_calls "
                "sample_rate_ppm validator_seed".split()
            ),
            label="slot audit policy",
        )
        slots = row["expected_slots"]
        if not isinstance(slots, list):
            raise SessionProtocolError("audit expected_slots must be an array")
        return cls(
            row["validator_seed"],
            row["sample_rate_ppm"],
            row["minimum_calls"],
            tuple(slots),
            row["expected_member_count"],
        )  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("optima.eval.slot-audit-policy.v1", self.to_dict())


_AUDIT_RECEIPT_FIELDS = frozenset(
    "baseline_refused compare_errors min_ratio mode n pid rank slot "
    "violations world_size worst_frac".split()
)


@dataclass(frozen=True)
class AuditReceiptFacts:
    """Strict raw scheduler-rank facts; no worker-produced verdict is accepted."""

    slot: str
    n: int
    violations: int
    baseline_refused: int
    compare_errors: int
    worst_frac: float
    min_ratio: float | None
    mode: str | None
    pid: int
    rank: int
    world_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.slot, str) or _TOKEN.fullmatch(self.slot) is None:
            raise SessionProtocolError("audit receipt slot is invalid")
        for name in ("n", "violations", "baseline_refused", "compare_errors"):
            object.__setattr__(
                self,
                name,
                _bounded_int(
                    getattr(self, name),
                    field_name=f"audit receipt {name}",
                    minimum=0,
                    maximum=1_000_000_000,
                ),
            )
        if self.violations > self.n:
            raise SessionProtocolError("audit violations exceed compared calls")
        object.__setattr__(
            self,
            "worst_frac",
            _bounded_float(
                self.worst_frac,
                field_name="audit receipt worst_frac",
                minimum=-1.0,
                maximum=1.0,
            ),
        )
        if self.min_ratio is not None:
            object.__setattr__(
                self,
                "min_ratio",
                _bounded_float(
                    self.min_ratio,
                    field_name="audit receipt min_ratio",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        if self.mode is not None and self.mode not in {
            "allclose", "cosine", "matched_ratio", "topk_overlap"
        }:
            raise SessionProtocolError("audit receipt mode is invalid")
        if self.n > 0 and (
            self.mode is None
            or self.min_ratio is None
            or ((self.violations == 0) != (self.worst_frac >= self.min_ratio))
        ):
            raise SessionProtocolError(
                "audit receipt comparison summary is internally inconsistent"
            )
        object.__setattr__(
            self,
            "pid",
            _bounded_int(
                self.pid,
                field_name="audit receipt pid",
                minimum=1,
                maximum=2_147_483_647,
            ),
        )
        object.__setattr__(
            self,
            "world_size",
            _bounded_int(
                self.world_size,
                field_name="audit receipt world_size",
                minimum=1,
                maximum=64,
            ),
        )
        object.__setattr__(
            self,
            "rank",
            _bounded_int(
                self.rank,
                field_name="audit receipt rank",
                minimum=0,
                maximum=self.world_size - 1,
            ),
        )

    @classmethod
    def from_receipt_dict(cls, value: object) -> "AuditReceiptFacts":
        row = _exact_object(
            value, fields=_AUDIT_RECEIPT_FIELDS, label="audit receipt"
        )
        return cls(**row)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, value: object) -> "AuditReceiptFacts":
        return cls.from_receipt_dict(value)

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in _AUDIT_RECEIPT_FIELDS}

    def to_gate_dict(self) -> dict[str, object]:
        return self.to_dict()


def _exact_object(value: object, *, fields: frozenset[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SessionProtocolError(f"{label} fields do not match the schema")
    return value


def _bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise SessionProtocolError(f"{field_name} must be boolean")
    return value


def _bounded_int(value: object, *, field_name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise SessionProtocolError(f"{field_name} must be an integer in [{minimum}, {maximum}]")
    return value


def _bounded_float(value: object, *, field_name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SessionProtocolError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise SessionProtocolError(f"{field_name} must be finite and in [{minimum}, {maximum}]")
    return result


def _optional_token(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise SessionProtocolError(f"{field_name} must be null or a bounded runtime token")
    return value


def _digest(value: object, *, field_name: str) -> str:
    return require_digest(value, field=field_name, error=SessionProtocolError)


def _binding_id(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or _HEX_128.fullmatch(value) is None
        or value == "0" * 32
    ):
        raise SessionProtocolError(f"{field_name} must be nonzero 128-bit lowercase hex")
    return value


def _validate_engine_kwargs(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SessionProtocolError("engine_config.engine_kwargs must be an object")
    unknown = set(value) - set(_ENGINE_KWARG_KINDS)
    if unknown:
        raise SessionProtocolError(
            "engine_config.engine_kwargs contains unsupported keys: "
            f"{sorted(unknown)!r}"
        )
    result: dict[str, object] = {}
    for key in sorted(value):
        item = value[key]
        kind = _ENGINE_KWARG_KINDS[key]
        field_name = f"engine_config.engine_kwargs.{key}"
        if kind == "bool":
            result[key] = _bool(item, field_name=field_name)
        elif kind == "positive_int":
            result[key] = _bounded_int(
                item, field_name=field_name, minimum=1, maximum=16_777_216
            )
        elif kind == "token":
            result[key] = _optional_token(item, field_name=field_name)
            if result[key] is None:
                raise SessionProtocolError(f"{field_name} must not be null")
        else:  # pragma: no cover - validator-owned table invariant
            raise AssertionError(f"unknown engine kwarg kind {kind!r}")
    return result


@dataclass(frozen=True)
class EngineSessionConfig:
    """Reviewed engine-construction inputs; never a workload or verdict policy."""

    model_path: str
    dtype: str
    deterministic: bool
    attention_backend: str | None
    disable_cuda_graph: bool
    mem_fraction_static: float
    log_level: str
    max_running_requests: int | None
    tp_size: int
    moe_runner_backend: str | None
    disable_custom_all_reduce: bool
    engine_kwargs: Mapping[str, object] = field(default_factory=dict)
    seam_bindings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.model_path != CONTAINER_MODEL_PATH:
            raise SessionProtocolError(f"engine_config.model_path must be {CONTAINER_MODEL_PATH!r}")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise SessionProtocolError("engine_config.dtype is unsupported")
        set_value = object.__setattr__
        set_value(self, "deterministic", _bool(
            self.deterministic, field_name="engine_config.deterministic"
        ))
        set_value(self, "attention_backend", _optional_token(
            self.attention_backend, field_name="engine_config.attention_backend"
        ))
        set_value(self, "disable_cuda_graph", _bool(
            self.disable_cuda_graph, field_name="engine_config.disable_cuda_graph"
        ))
        set_value(self, "mem_fraction_static", _bounded_float(
            self.mem_fraction_static, field_name="engine_config.mem_fraction_static",
            minimum=0.000001, maximum=0.999999,
        ))
        if not isinstance(self.log_level, str) or _TOKEN.fullmatch(self.log_level) is None:
            raise SessionProtocolError("engine_config.log_level is invalid")
        if self.max_running_requests is not None:
            set_value(self, "max_running_requests", _bounded_int(
                self.max_running_requests, field_name="engine_config.max_running_requests",
                minimum=1, maximum=1_048_576,
            ))
        set_value(self, "tp_size", _bounded_int(
            self.tp_size, field_name="engine_config.tp_size", minimum=1, maximum=64
        ))
        set_value(self, "moe_runner_backend", _optional_token(
            self.moe_runner_backend, field_name="engine_config.moe_runner_backend"
        ))
        set_value(self, "disable_custom_all_reduce", _bool(
            self.disable_custom_all_reduce,
            field_name="engine_config.disable_custom_all_reduce",
        ))
        set_value(self, "engine_kwargs", MappingProxyType(
            _validate_engine_kwargs(self.engine_kwargs)
        ))
        try:
            bindings = normalize_seam_bindings(self.seam_bindings)
        except ValueError as exc:
            raise SessionProtocolError(str(exc)) from exc
        set_value(self, "seam_bindings", bindings)

    def to_dict(self) -> dict[str, object]:
        row = {name: getattr(self, name) for name in ENGINE_CONFIG_FIELDS}
        row["engine_kwargs"] = dict(self.engine_kwargs)
        row["seam_bindings"] = list(self.seam_bindings)
        return row

    @property
    def digest(self) -> str:
        # Stack identities deliberately reject JSON floats.  Bind the exact
        # binary64 value through its shortest round-trippable decimal spelling.
        identity = self.to_dict()
        identity["mem_fraction_static"] = format(self.mem_fraction_static, ".17g")
        return canonical_digest("optima.eval.engine-session-config", identity)

    @classmethod
    def from_dict(cls, value: object) -> "EngineSessionConfig":
        row = _exact_object(value, fields=ENGINE_CONFIG_FIELDS, label="engine_config")
        values = dict(row)
        bindings = values.get("seam_bindings")
        if not isinstance(bindings, list):
            raise SessionProtocolError("engine_config.seam_bindings must be an array")
        values["seam_bindings"] = tuple(bindings)
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class RuntimePreflightFacts:
    """Raw live worker facts checked against validator-owned launch identity."""

    launch_digest: str
    runtime_digest: str
    stack_digest: str
    tree_digest: str
    engine_config_digest: str
    worker_distribution_digest: str
    model_revision_digest: str
    model_manifest_digest: str
    model_content_digest: str
    sglang_version: str
    gpu_architectures: tuple[str, ...]
    topology_digest: str
    loopback_only: bool
    read_only_inputs: bool
    private_writable_cache: bool

    def __post_init__(self) -> None:
        for name in (
            "launch_digest",
            "runtime_digest",
            "stack_digest",
            "tree_digest",
            "engine_config_digest",
            "worker_distribution_digest",
            "model_revision_digest",
            "model_manifest_digest",
            "model_content_digest",
            "topology_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), field_name=name))
        if (
            not isinstance(self.sglang_version, str)
            or _TOKEN.fullmatch(self.sglang_version) is None
        ):
            raise SessionProtocolError("preflight sglang_version is invalid")
        if (
            not isinstance(self.gpu_architectures, (tuple, list))
            or not 1 <= len(self.gpu_architectures) <= 64
            or any(
                not isinstance(value, str)
                or _ARCHITECTURE.fullmatch(value) is None
                for value in self.gpu_architectures
            )
        ):
            raise SessionProtocolError("preflight gpu_architectures is invalid")
        object.__setattr__(self, "gpu_architectures", tuple(self.gpu_architectures))
        for name in ("loopback_only", "read_only_inputs", "private_writable_cache"):
            value = _bool(getattr(self, name), field_name=f"preflight {name}")
            if not value:
                raise SessionProtocolError(f"preflight {name} is not proven")

    def to_dict(self) -> dict[str, object]:
        row = {name: getattr(self, name) for name in PREFLIGHT_FACT_FIELDS}
        row["gpu_architectures"] = list(self.gpu_architectures)
        return row

    @property
    def digest(self) -> str:
        return canonical_digest("optima.eval.runtime-preflight-facts", self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "RuntimePreflightFacts":
        row = _exact_object(
            value, fields=PREFLIGHT_FACT_FIELDS, label="runtime preflight facts"
        )
        values = dict(row)
        architectures = values.get("gpu_architectures")
        if not isinstance(architectures, list):
            raise SessionProtocolError("preflight gpu_architectures must be an array")
        values["gpu_architectures"] = tuple(architectures)
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class BatchRequest:
    """One host-disclosed prompt batch and its exact evidence shape."""

    session_id: str
    launch_digest: str
    request_id: str
    nonce: str
    batch_index: int
    prompts: tuple[str, ...]
    max_new_tokens: int
    top_logprobs_num: int
    temperature: float

    def __post_init__(self) -> None:
        for name in ("session_id", "request_id", "nonce"):
            object.__setattr__(self, name, _binding_id(getattr(self, name), field_name=name))
        if len({self.session_id, self.request_id, self.nonce}) != 3:
            raise SessionProtocolError("session_id, request_id, and nonce must be distinct")
        object.__setattr__(self, "launch_digest", _digest(
            self.launch_digest, field_name="launch_digest"
        ))
        object.__setattr__(self, "batch_index", _bounded_int(
            self.batch_index, field_name="batch_index", minimum=0,
            maximum=2_147_483_647,
        ))
        if (
            isinstance(self.prompts, (str, bytes))
            or not isinstance(self.prompts, Sequence)
            or not 1 <= len(self.prompts) <= MAX_PROMPTS_PER_BATCH
        ):
            raise SessionProtocolError("batch prompts count is invalid")
        clean: list[str] = []
        total_chars = 0
        for prompt in self.prompts:
            if not isinstance(prompt, str) or len(prompt) > MAX_PROMPT_CHARS:
                raise SessionProtocolError("batch contains an invalid/oversized prompt")
            total_chars += len(prompt)
            if total_chars > MAX_TOTAL_PROMPT_CHARS:
                raise SessionProtocolError("batch exceeds its total prompt-character bound")
            clean.append(prompt)
        object.__setattr__(self, "prompts", tuple(clean))
        object.__setattr__(self, "max_new_tokens", _bounded_int(
            self.max_new_tokens, field_name="max_new_tokens", minimum=1,
            maximum=MAX_NEW_TOKENS,
        ))
        object.__setattr__(self, "top_logprobs_num", _bounded_int(
            self.top_logprobs_num, field_name="top_logprobs_num", minimum=1,
            maximum=MAX_TOP_LOGPROBS,
        ))
        object.__setattr__(self, "temperature", _bounded_float(
            self.temperature, field_name="temperature", minimum=0.0, maximum=100.0
        ))
        expected_evidence_payload_bytes(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SESSION_SCHEMA, "type": "batch_request",
            "session_id": self.session_id, "launch_digest": self.launch_digest,
            "request_id": self.request_id, "nonce": self.nonce,
            "batch_index": self.batch_index, "prompts": list(self.prompts),
            "max_new_tokens": self.max_new_tokens,
            "top_logprobs_num": self.top_logprobs_num,
            "temperature": self.temperature,
        }


@dataclass(frozen=True)
class PromptEvidence:
    output_ids: tuple[int, ...]
    top_logprobs: tuple[tuple[tuple[float, int], ...], ...]


@dataclass(frozen=True)
class BatchEvidence:
    """Raw token/top-k facts returned by an isolated engine."""

    prompts: tuple[PromptEvidence, ...]

    @property
    def observed_tokens(self) -> int:
        return sum(len(prompt.output_ids) for prompt in self.prompts)


@dataclass
class _JSONBudget:
    remaining: int = MAX_JSON_ITEMS

    def take(self, count: int = 1) -> None:
        if count < 0 or count > self.remaining:
            raise SessionProtocolError("session JSON exceeds its item bound")
        self.remaining -= count


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SessionProtocolError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _walk_json(value: object, *, depth: int, budget: _JSONBudget) -> None:
    if depth > MAX_JSON_NESTING:
        raise SessionProtocolError("session JSON semantic nesting exceeds its bound")
    budget.take()
    if value is None or type(value) in (bool, str):
        return
    if type(value) is int:
        if not -(1 << 63) <= value < (1 << 63):
            raise SessionProtocolError("session JSON integer exceeds its bound")
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise SessionProtocolError("session JSON contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _walk_json(item, depth=depth + 1, budget=budget)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 256:
                raise SessionProtocolError("session JSON has an invalid object key")
            budget.take()
            _walk_json(item, depth=depth + 1, budget=budget)
        return
    raise SessionProtocolError("session JSON contains a non-JSON value")


def encode_message(message: Mapping[str, object], *, max_bytes: int) -> bytes:
    if not isinstance(message, Mapping):
        raise SessionProtocolError("session message must be an object")
    detached = dict(message)
    _walk_json(detached, depth=0, budget=_JSONBudget())
    try:
        data = json.dumps(
            detached,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SessionProtocolError(f"session message is not strict JSON: {exc}") from None
    if not data or len(data) > max_bytes:
        raise SessionProtocolError(f"session message exceeds {max_bytes} bytes")
    return data


def decode_message(data: bytes, *, max_bytes: int) -> dict[str, Any]:
    if not isinstance(data, bytes) or not data or len(data) > max_bytes:
        raise SessionProtocolError(f"session message exceeds {max_bytes} bytes")
    try:
        text = data.decode("utf-8", errors="strict")
        decoder = json.JSONDecoder(
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SessionProtocolError(f"non-finite JSON constant {value!r}")
            ),
        )
        value, end = decoder.raw_decode(text)
    except SessionProtocolError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise SessionProtocolError(f"session message is not valid JSON: {exc}") from None
    if end != len(text):
        raise SessionProtocolError("session message has trailing bytes/whitespace")
    if not isinstance(value, dict):
        raise SessionProtocolError("session message must decode to an object")
    _walk_json(value, depth=0, budget=_JSONBudget())
    return value


def frame_message(message: Mapping[str, object], *, max_bytes: int) -> bytes:
    payload = encode_message(message, max_bytes=max_bytes)
    return CONTROL_MAGIC + struct.pack(">I", len(payload)) + payload


def parse_frame_bytes(frame: bytes, *, max_bytes: int) -> dict[str, Any]:
    if not isinstance(frame, bytes) or len(frame) < FRAME_HEADER_BYTES:
        raise SessionProtocolError("session control frame is truncated")
    if frame[:4] != CONTROL_MAGIC:
        raise SessionProtocolError("session control frame magic/version mismatch")
    size = struct.unpack(">I", frame[4:8])[0]
    if size > max_bytes:
        raise SessionProtocolError("session control frame declares an oversized payload")
    if len(frame) != FRAME_HEADER_BYTES + size:
        raise SessionProtocolError("session control frame has trailing or missing bytes")
    return decode_message(frame[8:], max_bytes=max_bytes)


def make_init(
    config: EngineSessionConfig,
    *,
    session_id: str,
    launch_digest: str,
    expected_engine_config_digest: str,
    audit_policy: SlotAuditPolicy | None = None,
) -> dict[str, object]:
    if not isinstance(config, EngineSessionConfig):
        raise SessionProtocolError("init engine_config is not typed")
    expected = _digest(expected_engine_config_digest, field_name="expected_engine_config_digest")
    if config.digest != expected:
        raise SessionProtocolError("engine_config does not match its launch digest")
    message: dict[str, object] = {
        "engine_config": config.to_dict(),
        "engine_config_digest": expected,
        "launch_digest": _digest(launch_digest, field_name="launch_digest"),
        "schema": SESSION_SCHEMA,
        "session_id": _binding_id(session_id, field_name="session_id"),
        "type": "init",
    }
    if audit_policy is not None:
        if type(audit_policy) is not SlotAuditPolicy:
            raise SessionProtocolError("init audit_policy is not exactly typed")
        message["audit_policy"] = audit_policy.to_dict()
    return message


def audit_policy_from_init(message: object) -> SlotAuditPolicy | None:
    if not isinstance(message, Mapping):
        raise SessionProtocolError("init must be an object")
    if "audit_policy" not in message:
        return None
    return SlotAuditPolicy.from_dict(message["audit_policy"])


def validate_init(
    message: object,
    *,
    expected_launch_digest: str | None = None,
    expected_engine_config_digest: str | None = None,
    expected_audit_policy_digest: str | None = None,
) -> tuple[str, str, EngineSessionConfig]:
    base_fields = frozenset(
        "engine_config engine_config_digest launch_digest schema session_id type".split()
    )
    has_audit = isinstance(message, Mapping) and "audit_policy" in message
    fields = base_fields | ({"audit_policy"} if has_audit else set())
    row = _exact_object(message, fields=fields, label="init")
    if row["schema"] != SESSION_SCHEMA or row["type"] != "init":
        raise SessionProtocolError("init schema/type mismatch")
    session_id = _binding_id(row["session_id"], field_name="session_id")
    launch_digest = _digest(row["launch_digest"], field_name="launch_digest")
    config_digest = _digest(row["engine_config_digest"], field_name="engine_config_digest")
    config = EngineSessionConfig.from_dict(row["engine_config"])
    if config.digest != config_digest:
        raise SessionProtocolError("init engine_config digest mismatch")
    if expected_launch_digest is not None and launch_digest != _digest(
        expected_launch_digest, field_name="expected_launch_digest"
    ):
        raise SessionProtocolError("init launch binding is stale")
    if expected_engine_config_digest is not None and config_digest != _digest(
        expected_engine_config_digest, field_name="expected_engine_config_digest"
    ):
        raise SessionProtocolError("init engine_config binding is stale")
    audit_policy = audit_policy_from_init(row)
    if expected_audit_policy_digest is not None:
        expected_audit = _digest(
            expected_audit_policy_digest,
            field_name="expected_audit_policy_digest",
        )
        if audit_policy is None or audit_policy.digest != expected_audit:
            raise SessionProtocolError("init audit policy binding is stale")
    return session_id, launch_digest, config


def preflight_message(
    *, session_id: str, launch_digest: str, facts: RuntimePreflightFacts
) -> dict[str, object]:
    if not isinstance(facts, RuntimePreflightFacts):
        raise SessionProtocolError("runtime preflight facts are not typed")
    launch = _digest(launch_digest, field_name="launch_digest")
    if facts.launch_digest != launch:
        raise SessionProtocolError("runtime preflight launch binding mismatch")
    return {
        "facts": facts.to_dict(),
        "launch_digest": launch,
        "schema": SESSION_SCHEMA,
        "session_id": _binding_id(session_id, field_name="session_id"),
        "type": "preflight",
    }


def validate_preflight(
    message: object,
    *,
    session_id: str,
    launch_digest: str,
    expected_facts: RuntimePreflightFacts | None = None,
) -> RuntimePreflightFacts:
    fields = frozenset("facts launch_digest schema session_id type".split())
    row = _exact_object(message, fields=fields, label="preflight")
    exact = {
        "launch_digest": _digest(launch_digest, field_name="launch_digest"),
        "schema": SESSION_SCHEMA,
        "session_id": _binding_id(session_id, field_name="session_id"),
        "type": "preflight",
    }
    if any(row[name] != value for name, value in exact.items()):
        raise SessionProtocolError("runtime preflight binding is stale or malformed")
    facts = RuntimePreflightFacts.from_dict(row["facts"])
    if facts.launch_digest != exact["launch_digest"]:
        raise SessionProtocolError("runtime preflight facts bind another launch")
    if expected_facts is not None:
        if not isinstance(expected_facts, RuntimePreflightFacts):
            raise SessionProtocolError("expected runtime preflight facts are not typed")
        if facts != expected_facts:
            raise SessionProtocolError("runtime preflight facts differ from host policy")
    return facts


def preflight_accept_message(
    *, session_id: str, launch_digest: str, facts: RuntimePreflightFacts
) -> dict[str, object]:
    """Authorize engine entry only after the host accepted exact live facts."""

    if not isinstance(facts, RuntimePreflightFacts):
        raise SessionProtocolError("accepted runtime preflight facts are not typed")
    launch = _digest(launch_digest, field_name="launch_digest")
    if facts.launch_digest != launch:
        raise SessionProtocolError("preflight acceptance launch binding mismatch")
    return {
        "facts_digest": facts.digest,
        "launch_digest": launch,
        "schema": SESSION_SCHEMA,
        "session_id": _binding_id(session_id, field_name="session_id"),
        "type": "preflight_accept",
    }


def validate_preflight_accept(
    message: object,
    *,
    session_id: str,
    launch_digest: str,
    expected_facts_digest: str,
) -> None:
    expected = {
        "facts_digest": _digest(
            expected_facts_digest, field_name="expected_facts_digest"
        ),
        "launch_digest": _digest(launch_digest, field_name="launch_digest"),
        "schema": SESSION_SCHEMA,
        "session_id": _binding_id(session_id, field_name="session_id"),
        "type": "preflight_accept",
    }
    if message != expected:
        raise SessionProtocolError("preflight acceptance is stale or malformed")


def ready_message(
    *,
    session_id: str,
    launch_digest: str,
    discovery_activation: DiscoveryActivationReceipt | None = None,
) -> dict[str, object]:
    message: dict[str, object] = {
        "launch_digest": _digest(launch_digest, field_name="launch_digest"),
        "schema": SESSION_SCHEMA,
        "session_id": _binding_id(session_id, field_name="session_id"),
        "type": "ready",
    }
    if discovery_activation is not None:
        from optima.discovery_overlay import DiscoveryActivationReceipt

        if type(discovery_activation) is not DiscoveryActivationReceipt:
            raise SessionProtocolError(
                "discovery ready activation is not a typed receipt"
            )
        message["discovery_activation"] = discovery_activation.to_dict()
    return message


def validate_ready(
    message: object,
    *,
    session_id: str,
    launch_digest: str,
    expected_discovery_identity_digest: str | None = None,
    expected_discovery_tp_size: int | None = None,
    expected_discovery_sglang_version: str | None = None,
) -> DiscoveryActivationReceipt | None:
    expected_values = (
        expected_discovery_identity_digest,
        expected_discovery_tp_size,
        expected_discovery_sglang_version,
    )
    if all(value is None for value in expected_values):
        if message != ready_message(
            session_id=session_id, launch_digest=launch_digest
        ):
            raise SessionProtocolError(
                "worker ready marker is early, stale, or malformed"
            )
        return None
    if any(value is None for value in expected_values):
        raise SessionProtocolError(
            "discovery ready expectation is incomplete"
        )
    identity = _digest(
        expected_discovery_identity_digest,
        field_name="expected_discovery_identity_digest",
    )
    tp_size = _bounded_int(
        expected_discovery_tp_size,
        field_name="expected_discovery_tp_size",
        minimum=1,
        maximum=64,
    )
    version = expected_discovery_sglang_version
    if not isinstance(version, str) or _TOKEN.fullmatch(version) is None:
        raise SessionProtocolError(
            "expected discovery SGLang version is invalid"
        )
    fields = frozenset(
        "discovery_activation launch_digest schema session_id type".split()
    )
    row = _exact_object(message, fields=fields, label="discovery ready")
    expected_envelope = ready_message(
        session_id=session_id, launch_digest=launch_digest
    )
    if any(row[name] != value for name, value in expected_envelope.items()):
        raise SessionProtocolError("worker ready marker is early, stale, or malformed")
    from optima.discovery_overlay import (
        DiscoveryActivationReceipt,
        DiscoveryOverlayActivationError,
    )

    try:
        receipt = DiscoveryActivationReceipt.from_dict(
            row["discovery_activation"]
        )
    except DiscoveryOverlayActivationError as exc:
        raise SessionProtocolError(
            f"discovery ready activation is malformed: {exc}"
        ) from None
    if (
        receipt.overlay_identity_digest != identity
        or receipt.tp_size != tp_size
        or receipt.driver_origin.version != version
    ):
        raise SessionProtocolError(
            "discovery ready activation differs from host policy"
        )
    return receipt


_BATCH_REQUEST_FIELDS = frozenset("""
batch_index launch_digest max_new_tokens nonce prompts request_id schema session_id
temperature top_logprobs_num type
""".split())


def batch_request(
    *,
    session_id: str,
    launch_digest: str,
    request_id: str,
    nonce: str,
    batch_index: int,
    prompts: Sequence[str],
    max_new_tokens: int,
    top_logprobs_num: int,
    temperature: float,
) -> dict[str, object]:
    return BatchRequest(
        session_id, launch_digest, request_id, nonce, batch_index, tuple(prompts),
        max_new_tokens, top_logprobs_num, temperature,
    ).to_dict()


def validate_batch_request(message: object) -> BatchRequest:
    row = _exact_object(message, fields=_BATCH_REQUEST_FIELDS, label="batch request")
    if row["schema"] != SESSION_SCHEMA or row["type"] != "batch_request":
        raise SessionProtocolError("batch request schema/type mismatch")
    prompts = row["prompts"]
    if not isinstance(prompts, list):
        raise SessionProtocolError("batch request prompts must be an array")
    return BatchRequest(
        row["session_id"], row["launch_digest"], row["request_id"], row["nonce"],
        row["batch_index"], tuple(prompts), row["max_new_tokens"],
        row["top_logprobs_num"], row["temperature"],
    )  # type: ignore[arg-type]


_AUDIT_EVIDENCE_FIELDS = frozenset(
    "audit_policy_digest batch_index launch_digest nonce receipts request_id "
    "schema session_id type".split()
)


def audit_evidence_message(
    *,
    request: BatchRequest,
    policy: SlotAuditPolicy,
    receipts: Sequence[AuditReceiptFacts],
) -> dict[str, object]:
    if type(request) is not BatchRequest or type(policy) is not SlotAuditPolicy:
        raise SessionProtocolError("audit evidence binding is not exactly typed")
    facts = tuple(receipts)
    if len(facts) > MAX_AUDIT_RECEIPTS or any(
        type(row) is not AuditReceiptFacts for row in facts
    ):
        raise SessionProtocolError("audit evidence receipt sequence is invalid")
    return {
        "audit_policy_digest": policy.digest,
        "batch_index": request.batch_index,
        "launch_digest": request.launch_digest,
        "nonce": request.nonce,
        "receipts": [row.to_dict() for row in facts],
        "request_id": request.request_id,
        "schema": SESSION_SCHEMA,
        "session_id": request.session_id,
        "type": "audit_evidence",
    }


def validate_audit_evidence(
    message: object,
    *,
    request: BatchRequest,
    policy: SlotAuditPolicy,
) -> tuple[AuditReceiptFacts, ...]:
    if type(request) is not BatchRequest or type(policy) is not SlotAuditPolicy:
        raise SessionProtocolError("audit evidence expectation is not exactly typed")
    row = _exact_object(
        message, fields=_AUDIT_EVIDENCE_FIELDS, label="audit evidence"
    )
    expected = {
        "audit_policy_digest": policy.digest,
        "batch_index": request.batch_index,
        "launch_digest": request.launch_digest,
        "nonce": request.nonce,
        "request_id": request.request_id,
        "schema": SESSION_SCHEMA,
        "session_id": request.session_id,
        "type": "audit_evidence",
    }
    if any(row[name] != value for name, value in expected.items()):
        raise SessionProtocolError(
            "audit evidence nonce/request/session/launch/policy binding mismatch"
        )
    raw_receipts = row["receipts"]
    if not isinstance(raw_receipts, list) or len(raw_receipts) > MAX_AUDIT_RECEIPTS:
        raise SessionProtocolError("audit evidence receipts must be a bounded array")
    receipts = tuple(AuditReceiptFacts.from_dict(value) for value in raw_receipts)
    if len({(row.slot, row.rank) for row in receipts}) != len(receipts):
        raise SessionProtocolError("audit evidence duplicates one slot/rank receipt")
    return receipts


_EVIDENCE_BINDING = struct.Struct(">16s32s16s16sIIIH2x")
_TOKEN_ID = struct.Struct(">I")
_TOPK_ENTRY = struct.Struct(">fI")


def expected_evidence_payload_bytes(request: BatchRequest) -> int:
    if not isinstance(request, BatchRequest):
        raise SessionProtocolError("evidence request is not typed")
    prompt_count = len(request.prompts)
    per_position = _TOKEN_ID.size + request.top_logprobs_num * _TOPK_ENTRY.size
    total = _EVIDENCE_BINDING.size + prompt_count * request.max_new_tokens * per_position
    if total > MAX_BATCH_RESPONSE_BYTES:
        raise SessionProtocolError("exact binary evidence exceeds its hard bound")
    return total


def _validated_evidence(evidence: BatchEvidence, *, request: BatchRequest) -> BatchEvidence:
    if not isinstance(evidence, BatchEvidence) or len(evidence.prompts) != len(request.prompts):
        raise SessionProtocolError("binary evidence prompt count is invalid")
    clean_prompts: list[PromptEvidence] = []
    for prompt in evidence.prompts:
        if not isinstance(prompt, PromptEvidence):
            raise SessionProtocolError("binary prompt evidence is not typed")
        if len(prompt.output_ids) != request.max_new_tokens:
            raise SessionProtocolError("binary evidence returned a short/oversized output")
        if len(prompt.top_logprobs) != request.max_new_tokens:
            raise SessionProtocolError("binary evidence has wrong top-k position count")
        clean_ids = [
            _bounded_int(token, field_name="output token ID", minimum=0,
                         maximum=2_147_483_647)
            for token in prompt.output_ids
        ]
        clean_positions: list[tuple[tuple[float, int], ...]] = []
        for position in prompt.top_logprobs:
            if not isinstance(position, (tuple, list)) or len(position) != request.top_logprobs_num:
                raise SessionProtocolError("binary evidence top-k width is invalid")
            clean_position: list[tuple[float, int]] = []
            seen: set[int] = set()
            for entry in position:
                if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                    raise SessionProtocolError("binary top-k entry is invalid")
                logprob = _bounded_float(entry[0], field_name="top-k logprob",
                                         minimum=-1_000_000.0, maximum=1e-4)
                token_id = _bounded_int(entry[1], field_name="top-k token ID",
                                        minimum=0, maximum=2_147_483_647)
                if token_id in seen:
                    raise SessionProtocolError("binary top-k token ID is duplicated")
                seen.add(token_id)
                clean_position.append((logprob, token_id))
            if any(a[0] < b[0] for a, b in zip(clean_position, clean_position[1:])):
                raise SessionProtocolError("binary top-k entries are not descending")
            if sum(math.exp(entry[0]) for entry in clean_position) > 1.0001:
                raise SessionProtocolError("binary top-k probability mass exceeds one")
            clean_positions.append(tuple(clean_position))
        clean_prompts.append(PromptEvidence(tuple(clean_ids), tuple(clean_positions)))
    return BatchEvidence(tuple(clean_prompts))


def evidence_frame(evidence: BatchEvidence, *, request: BatchRequest) -> bytes:
    clean = _validated_evidence(evidence, request=request)
    payload = bytearray(_EVIDENCE_BINDING.pack(
        bytes.fromhex(request.session_id), bytes.fromhex(request.launch_digest),
        bytes.fromhex(request.request_id), bytes.fromhex(request.nonce),
        request.batch_index, len(request.prompts), request.max_new_tokens,
        request.top_logprobs_num,
    ))
    for prompt in clean.prompts:
        for token_id, position in zip(prompt.output_ids, prompt.top_logprobs, strict=True):
            payload.extend(_TOKEN_ID.pack(token_id))
            for logprob, top_token_id in position:
                payload.extend(_TOPK_ENTRY.pack(logprob, top_token_id))
    expected = expected_evidence_payload_bytes(request)
    if len(payload) != expected:  # pragma: no cover - format-table invariant
        raise AssertionError("binary evidence encoder violated its exact size")
    return EVIDENCE_MAGIC + struct.pack(">I", expected) + bytes(payload)


def decode_evidence_payload(payload: bytes, *, request: BatchRequest) -> BatchEvidence:
    expected = expected_evidence_payload_bytes(request)
    if not isinstance(payload, bytes) or len(payload) != expected:
        raise SessionProtocolError("binary evidence has the wrong exact size")
    try:
        (session, launch, request_id, nonce, batch_index, prompt_count,
         token_count, topk_width) = _EVIDENCE_BINDING.unpack_from(payload)
    except struct.error:
        raise SessionProtocolError("binary evidence binding is truncated") from None
    if (
        session.hex() != request.session_id
        or launch.hex() != request.launch_digest
        or request_id.hex() != request.request_id
        or nonce.hex() != request.nonce
        or batch_index != request.batch_index
        or prompt_count != len(request.prompts)
        or token_count != request.max_new_tokens
        or topk_width != request.top_logprobs_num
    ):
        raise SessionProtocolError("binary evidence nonce/request/session/launch binding mismatch")
    offset = _EVIDENCE_BINDING.size
    prompts: list[PromptEvidence] = []
    for _ in range(prompt_count):
        output_ids: list[int] = []
        positions: list[tuple[tuple[float, int], ...]] = []
        for _ in range(token_count):
            (token_id,) = _TOKEN_ID.unpack_from(payload, offset)
            offset += _TOKEN_ID.size
            output_ids.append(token_id)
            position: list[tuple[float, int]] = []
            for _ in range(topk_width):
                logprob, top_token_id = _TOPK_ENTRY.unpack_from(payload, offset)
                offset += _TOPK_ENTRY.size
                position.append((float(logprob), top_token_id))
            positions.append(tuple(position))
        prompts.append(PromptEvidence(tuple(output_ids), tuple(positions)))
    if offset != len(payload):  # pragma: no cover - exact size already proves it
        raise SessionProtocolError("binary evidence contains trailing bytes")
    return _validated_evidence(BatchEvidence(tuple(prompts)), request=request)


def parse_evidence_frame_bytes(frame: bytes, *, request: BatchRequest) -> BatchEvidence:
    if not isinstance(frame, bytes) or len(frame) < FRAME_HEADER_BYTES:
        raise SessionProtocolError("binary evidence frame is truncated")
    if frame[:4] != EVIDENCE_MAGIC:
        raise SessionProtocolError("binary evidence frame magic/version mismatch")
    size = struct.unpack(">I", frame[4:8])[0]
    expected = expected_evidence_payload_bytes(request)
    if size != expected:
        raise SessionProtocolError("binary evidence frame declares the wrong exact size")
    if len(frame) != FRAME_HEADER_BYTES + size:
        raise SessionProtocolError("binary evidence frame has trailing or missing bytes")
    return decode_evidence_payload(frame[8:], request=request)


def error_message(
    *,
    session_id: str,
    launch_digest: str,
    stage: str,
    error: BaseException,
    request: BatchRequest | None = None,
) -> dict[str, object]:
    if not isinstance(stage, str) or not stage or len(stage) > 128 or not _TOKEN.fullmatch(stage):
        raise SessionProtocolError("worker error stage is invalid")
    if request is not None and (
        request.session_id != session_id or request.launch_digest != launch_digest
    ):
        raise SessionProtocolError("worker error request binding mismatch")
    first_arg = error.args[0] if error.args else None
    if not error.args:
        bounded_message = ""
    elif type(first_arg) is str:
        bounded_message = first_arg[:MAX_ERROR_CHARS]
    elif type(first_arg) in {int, float, bool, type(None)}:
        bounded_message = str(first_arg)[:MAX_ERROR_CHARS]
    else:
        # Candidate-defined exception rendering is executable code and can return
        # an unbounded value. Preserve the type/stage binding without invoking it.
        bounded_message = "<non-primitive exception detail omitted>"
    return {
        "batch_index": None if request is None else request.batch_index,
        "error_type": type(error).__name__[:128],
        "launch_digest": _digest(launch_digest, field_name="launch_digest"),
        "message": bounded_message,
        "nonce": None if request is None else request.nonce,
        "request_id": None if request is None else request.request_id,
        "schema": SESSION_SCHEMA,
        "session_id": _binding_id(session_id, field_name="session_id"),
        "stage": stage,
        "type": "session_error",
    }


def parse_error_message(
    message: object,
    *,
    session_id: str,
    launch_digest: str,
    request: BatchRequest | None = None,
) -> tuple[str, str, str] | None:
    if not isinstance(message, Mapping) or (
        message.get("schema") != SESSION_SCHEMA
        or message.get("type") != "session_error"
    ):
        return None
    fields = frozenset("""
    batch_index error_type launch_digest message nonce request_id schema session_id stage type
    """.split())
    row = _exact_object(message, fields=fields, label="worker error")
    exact_request = {
        "batch_index": None if request is None else request.batch_index,
        "nonce": None if request is None else request.nonce,
        "request_id": None if request is None else request.request_id,
    }
    if (
        row["session_id"] != _binding_id(session_id, field_name="session_id")
        or row["launch_digest"] != _digest(launch_digest, field_name="launch_digest")
        or any(row[name] != value for name, value in exact_request.items())
    ):
        raise SessionProtocolError("worker error marker has a stale binding")
    for name, maximum in (
        ("stage", 128),
        ("error_type", 128),
        ("message", MAX_ERROR_CHARS),
    ):
        value = row[name]
        if not isinstance(value, str) or len(value) > maximum or "\x00" in value:
            raise SessionProtocolError("worker error marker is malformed")
    if not row["stage"] or not _TOKEN.fullmatch(row["stage"]):
        raise SessionProtocolError("worker error stage is invalid")
    return row["stage"], row["error_type"], row["message"]  # type: ignore[return-value]
