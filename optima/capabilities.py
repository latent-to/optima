"""Canonical live-call descriptors and declarative specialization domains.

The bundle metadata describes *where* a candidate implementation is valid.  The
validator's arena binding describes the call that is actually happening.  This
module is the narrow, data-only boundary between those two facts:

``CallDescriptor``
    A canonical mapping populated by validator-owned integration code.  It has
    no tensors or miner callables, so it is safe to construct before selecting a
    candidate.

``CapabilityDomain``
    A conjunction of exact, enumerated, and inclusive-range predicates parsed
    from the bundle's ``capabilities`` metadata object.

Missing descriptor fields are mismatches, not wildcards.  This is deliberate:
if a miner specializes for ``head_dim=128`` but an arena binding does not report
``head_dim``, the validator serves stock rather than guessing that the kernel is
safe.  Unknown predicate names are rejected at metadata-load time so a typo does
not turn into a silently unreachable submission.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any


CapabilityScalar = str | int | bool


class CapabilityMetadataError(ValueError):
    """Raised when normative capability metadata is malformed."""


# Validator-owned vocabulary.  Adding a field means an arena binding can
# populate it with stable semantics; bundle metadata may not invent fields.
# Keep shape/count values numeric and runtime/model properties textual.
NUMERIC_FIELDS = frozenset(
    {
        "alignment",
        "batch_size",
        "block_size",
        "ep_size",
        "exp_tokens",
        "head_dim",
        "hidden_dim",
        "intermediate_dim",
        "kv_len",
        "last_dim",
        "num_experts",
        "num_kv_heads",
        "num_q_heads",
        "num_tokens",
        "page_size",
        "q_len",
        "top_k",
        "tp_size",
        "world_size",
    }
)

CONTEXT_FIELDS = frozenset(
    {
        "architecture",
        "dtype",
        "graph_mode",
        "layout",
        "model",
        "phase",
        "quant",
        "runtime",
    }
)

SUPPORTED_FIELDS = NUMERIC_FIELDS | CONTEXT_FIELDS

# Canonical memory-layout name for the MSA prefill score-call ABI.  Q and the
# validator-gathered index-K are dense row-major matrices.  The score output is a
# non-overlapping row-major view whose row pitch may be padded by the enclosing
# batch slab (a contiguous view is also legal).  Both offline verify and the live
# sglang binding use this exact descriptor value.
MSA_PREFILL_ROW_MAJOR_LAYOUT = "row_major"

_FIELD_ALIASES = {
    "arch": "architecture",
    "dtype_name": "dtype",
    "ep": "ep_size",
    "tp": "tp_size",
    "world": "world_size",
    "worldsize": "world_size",
}
_FIELD_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def canonical_field_name(name: str) -> str:
    """Return the canonical descriptor key for a validator-provided name."""
    if not isinstance(name, str):
        raise TypeError(f"capability field name must be str, got {type(name).__name__}")
    field = name.strip().lower().replace("-", "_")
    field = _FIELD_ALIASES.get(field, field)
    if not _FIELD_RE.fullmatch(field):
        raise ValueError(f"invalid capability field name: {name!r}")
    return field


_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
    "torch.bfloat16": "bfloat16",
    "torch.float16": "float16",
    "torch.float32": "float32",
}
_GRAPH_MODE_ALIASES = {
    "capture": "cuda_graph",
    "capturing": "cuda_graph",
    "cuda-graph": "cuda_graph",
    "graph": "cuda_graph",
    "graphs_on": "cuda_graph",
    "on": "cuda_graph",
    "replay": "cuda_graph",
    "graphs_off": "eager",
    "off": "eager",
}
_PHASE_ALIASES = {"extend": "prefill"}
_QUANT_ALIASES = {"none": "dense", "unquantized": "dense"}


def _canonical_text(field: str, value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field} must not be empty")
    if field == "model":
        # Model identifiers can be case-sensitive repository names.
        return value
    token = value.lower().replace("-", "_")
    if field == "dtype":
        return _DTYPE_ALIASES.get(value.lower(), token)
    if field == "architecture":
        # Accept common spellings (SM_103 / sm-103) but preserve suffixes such
        # as ``a`` in architecture targets.
        return token.replace("sm_", "sm", 1) if token.startswith("sm_") else token
    if field == "graph_mode":
        return _GRAPH_MODE_ALIASES.get(token, token)
    if field == "phase":
        return _PHASE_ALIASES.get(token, token)
    if field == "quant":
        return _QUANT_ALIASES.get(token, token)
    return token


def canonical_value(field: str, value: Any) -> CapabilityScalar:
    """Type-check and normalize one descriptor/predicate value."""
    field = canonical_field_name(field)
    if field in NUMERIC_FIELDS:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be an integer, got {value!r}")
        if value < 0:
            raise ValueError(f"{field} must be non-negative, got {value}")
        return int(value)
    if field in CONTEXT_FIELDS:
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a string, got {value!r}")
        return _canonical_text(field, value)
    # CallDescriptor accepts future validator-private fields so an arena can be
    # developed before the public metadata vocabulary expands.  Metadata parsing
    # separately rejects fields outside SUPPORTED_FIELDS.
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"unsupported value for descriptor field {field!r}: {value!r}")


@dataclass(frozen=True, init=False)
class CallDescriptor(Mapping[str, CapabilityScalar]):
    """Immutable canonical description of one live validator call.

    Construct with a mapping and/or keyword fields.  Aliases such as ``arch``,
    ``dtype_name``, and ``tp`` are normalized.  Supplying conflicting aliases is
    rejected rather than allowing call-site ordering to change dispatch.
    ``None`` means "unknown" and is omitted; a predicate for that field will then
    fail closed.
    """

    _items: tuple[tuple[str, CapabilityScalar], ...]

    def __init__(
        self,
        values: Mapping[str, Any] | None = None,
        /,
        **fields: Any,
    ) -> None:
        merged: dict[str, CapabilityScalar] = {}
        raw_items = list((values or {}).items()) + list(fields.items())
        for raw_name, raw_value in raw_items:
            if raw_value is None:
                continue
            name = canonical_field_name(raw_name)
            value = canonical_value(name, raw_value)
            if name in merged and merged[name] != value:
                raise ValueError(
                    f"conflicting values for canonical descriptor field {name!r}: "
                    f"{merged[name]!r} vs {value!r}"
                )
            merged[name] = value
        object.__setattr__(self, "_items", tuple(sorted(merged.items())))

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | None = None, /, **fields: Any
    ) -> "CallDescriptor":
        return cls(values, **fields)

    @classmethod
    def from_legacy(
        cls,
        *,
        dtype_name: str,
        last_dim: int,
        arch: str | None,
        num_tokens: int | None = None,
    ) -> "CallDescriptor":
        """Bridge for existing dispatchers while they migrate field-by-field."""
        return cls(
            dtype=dtype_name,
            last_dim=last_dim,
            architecture=arch,
            num_tokens=num_tokens,
        )

    def __getitem__(self, key: str) -> CapabilityScalar:
        key = canonical_field_name(key)
        for name, value in self._items:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def as_dict(self) -> dict[str, CapabilityScalar]:
        return dict(self._items)

    def with_updates(self, **fields: Any) -> "CallDescriptor":
        values = self.as_dict()
        for raw_name, raw_value in fields.items():
            name = canonical_field_name(raw_name)
            if raw_value is None:
                values.pop(name, None)
            else:
                values[name] = canonical_value(name, raw_value)
        return CallDescriptor(values)


def collective_call_descriptor(
    *,
    dtype: str,
    architecture: str | None,
    graph_mode: str,
    world_size: int,
    tp_size: int | None = None,
    model: str | None = None,
    quant: str = "dense",
    layout: str = "row_major",
    dimensions: Mapping[str, Any] | None = None,
) -> CallDescriptor:
    """Canonical descriptor shared by distributed verify and live bindings.

    The caller owns semantic dimension extraction; this helper owns the stable
    collective context vocabulary. ``world_size`` is the size of the actual process
    group handed to the candidate, and TP defaults to that same group rather than a
    caller-provided environment hint.
    """

    fields: dict[str, Any] = {
        "dtype": dtype,
        "architecture": architecture,
        "graph_mode": graph_mode,
        "layout": layout,
        "quant": quant,
        "model": model,
        "world_size": world_size,
        "tp_size": world_size if tp_size is None else tp_size,
    }
    fields.update(dimensions or {})
    return CallDescriptor(fields)


def msa_prefill_call_descriptor(
    *,
    dtype: str,
    architecture: str | None,
    head_dim: int,
    block_size: int,
    q_len: int,
    kv_len: int,
    top_k: int,
    num_kv_heads: int = 1,
    tp_size: int | None = None,
    world_size: int | None = None,
) -> CallDescriptor:
    """Describe one canonical ``attention.msa_prefill_block_score`` call.

    The serving seam invokes the candidate once per request and query head, after
    gathering that request's index-K.  Consequently ``batch_size`` and
    ``num_q_heads`` are one here even when the surrounding engine batch contains
    many requests/heads.  This distinction is load-bearing for q-length variants:
    miners specialize against the tensor call they actually receive, not an
    unrelated aggregate batch shape.
    """

    return CallDescriptor(
        dtype=dtype,
        architecture=architecture,
        last_dim=head_dim,
        num_tokens=q_len,
        head_dim=head_dim,
        block_size=block_size,
        q_len=q_len,
        kv_len=kv_len,
        batch_size=1,
        num_q_heads=1,
        num_kv_heads=num_kv_heads,
        top_k=top_k,
        phase="prefill",
        layout=MSA_PREFILL_ROW_MAJOR_LAYOUT,
        graph_mode="eager",
        quant="dense",
        tp_size=tp_size,
        world_size=world_size,
    )


@dataclass(frozen=True)
class CapabilityPredicate:
    """One field constraint: allowed values or an inclusive numeric range."""

    field: str
    allowed: tuple[CapabilityScalar, ...] = ()
    minimum: int | None = None
    maximum: int | None = None

    def accepts(self, actual: CapabilityScalar) -> bool:
        if self.allowed:
            return actual in self.allowed
        if isinstance(actual, bool) or not isinstance(actual, int):
            return False
        if self.minimum is not None and actual < self.minimum:
            return False
        if self.maximum is not None and actual > self.maximum:
            return False
        return True

    @property
    def expected(self) -> str:
        if self.allowed:
            if len(self.allowed) == 1:
                return f"exactly {self.allowed[0]!r}"
            return f"one of {list(self.allowed)!r}"
        lo = "-inf" if self.minimum is None else str(self.minimum)
        hi = "+inf" if self.maximum is None else str(self.maximum)
        return f"in [{lo}, {hi}]"


@dataclass(frozen=True)
class CapabilityMismatch:
    field: str
    reason: str
    expected: str
    actual: CapabilityScalar | None = None


@dataclass(frozen=True)
class CapabilityMatch:
    mismatches: tuple[CapabilityMismatch, ...] = ()

    @property
    def accepted(self) -> bool:
        return not self.mismatches

    def __bool__(self) -> bool:
        return self.accepted


@dataclass(frozen=True)
class CapabilityDomain:
    """Conjunction of miner-declared, validator-parsed predicates."""

    predicates: tuple[CapabilityPredicate, ...] = ()

    def match(self, descriptor: CallDescriptor) -> CapabilityMatch:
        mismatches: list[CapabilityMismatch] = []
        for predicate in self.predicates:
            if predicate.field not in descriptor:
                mismatches.append(
                    CapabilityMismatch(
                        field=predicate.field,
                        reason="missing",
                        expected=predicate.expected,
                    )
                )
                continue
            actual = descriptor[predicate.field]
            if not predicate.accepts(actual):
                mismatches.append(
                    CapabilityMismatch(
                        field=predicate.field,
                        reason="outside_domain",
                        expected=predicate.expected,
                        actual=actual,
                    )
                )
        return CapabilityMatch(tuple(mismatches))

    @property
    def constrained_fields(self) -> frozenset[str]:
        return frozenset(predicate.field for predicate in self.predicates)


_PREDICATE_KEYS = frozenset({"exact", "one_of", "min", "max"})


def _parse_predicate(field: str, raw: Any) -> CapabilityPredicate:
    field = canonical_field_name(field)
    if field not in SUPPORTED_FIELDS:
        raise CapabilityMetadataError(
            f"unsupported capability field {field!r}; validator supports "
            f"{sorted(SUPPORTED_FIELDS)}"
        )

    # Concise spellings are intentional conveniences for bundle metadata:
    # scalar -> exact, list -> one_of, object -> the fully explicit schema.
    if not isinstance(raw, (dict, list, tuple)):
        raw = {"exact": raw}
    elif isinstance(raw, (list, tuple)):
        raw = {"one_of": raw}

    unknown = set(raw) - _PREDICATE_KEYS
    if unknown:
        raise CapabilityMetadataError(
            f"capabilities.{field} has unknown keys: {sorted(unknown)}"
        )
    has_allowed = "exact" in raw or "one_of" in raw
    has_range = "min" in raw or "max" in raw
    if has_allowed and has_range:
        raise CapabilityMetadataError(
            f"capabilities.{field} cannot mix exact/one_of with min/max"
        )
    if "exact" in raw and "one_of" in raw:
        raise CapabilityMetadataError(
            f"capabilities.{field} cannot set both exact and one_of"
        )
    if not has_allowed and not has_range:
        raise CapabilityMetadataError(f"capabilities.{field} has no predicate")

    try:
        if has_allowed:
            values = raw["one_of"] if "one_of" in raw else [raw["exact"]]
            if not isinstance(values, (list, tuple)) or not values:
                raise CapabilityMetadataError(
                    f"capabilities.{field}.one_of must be a non-empty list"
                )
            allowed: list[CapabilityScalar] = []
            for item in values:
                value = canonical_value(field, item)
                if value not in allowed:
                    allowed.append(value)
            return CapabilityPredicate(field=field, allowed=tuple(allowed))

        if field not in NUMERIC_FIELDS:
            raise CapabilityMetadataError(
                f"capabilities.{field} is contextual and cannot use min/max"
            )
        minimum = canonical_value(field, raw["min"]) if "min" in raw else None
        maximum = canonical_value(field, raw["max"]) if "max" in raw else None
        assert minimum is None or isinstance(minimum, int)
        assert maximum is None or isinstance(maximum, int)
        if minimum is not None and maximum is not None and minimum > maximum:
            raise CapabilityMetadataError(
                f"capabilities.{field} has min {minimum} greater than max {maximum}"
            )
        return CapabilityPredicate(field=field, minimum=minimum, maximum=maximum)
    except CapabilityMetadataError:
        raise
    except (TypeError, ValueError) as exc:
        raise CapabilityMetadataError(f"invalid capabilities.{field}: {exc}") from exc


def capability_domain_from_metadata(meta: Mapping[str, Any] | None) -> CapabilityDomain:
    """Parse the normative ``capabilities`` object from one op metadata JSON.

    Existing top-level eligibility keys and descriptive ``regime`` objects are
    deliberately not reinterpreted here; :func:`eligibility_from_metadata` keeps
    handling the former, while the latter remains non-normative documentation.
    """
    if meta is None:
        meta = {}
    if not isinstance(meta, Mapping):
        raise CapabilityMetadataError("eligibility metadata must be a JSON object")
    raw = meta.get("capabilities")
    if raw is None:
        return CapabilityDomain()
    if not isinstance(raw, Mapping):
        raise CapabilityMetadataError("metadata 'capabilities' must be an object")
    predicates: list[CapabilityPredicate] = []
    seen: set[str] = set()
    for raw_field, spec in raw.items():
        try:
            field = canonical_field_name(raw_field)
        except (TypeError, ValueError) as exc:
            raise CapabilityMetadataError(str(exc)) from exc
        if field in seen:
            raise CapabilityMetadataError(
                f"duplicate canonical capability field {field!r}"
            )
        seen.add(field)
        predicates.append(_parse_predicate(field, spec))
    predicates.sort(key=lambda predicate: predicate.field)
    return CapabilityDomain(tuple(predicates))
