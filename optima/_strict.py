"""Shared strict-validation kernel.

One audited body per validation concept. Every validator raises the CALLER's
module error class (the ``error=`` parameter), so modules keep their typed
exception APIs without re-implementing the checks. Per-module copies of these
helpers are exactly the drift this module exists to end: 26 hand-rolled
``_digest`` shims disagreed on whether the all-zero digest is admissible.

Policy stays local, mechanism lives here: an identifier's grammar or an
integer's bounds are the calling module's decision (passed in explicitly);
the check-and-raise body is shared and audited once.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Mapping
from typing import Any

from .stack_identity import require_sha256_hex

_ZERO_DIGEST = "0" * 64


def require_digest(
    value: object,
    *,
    field: str,
    error: type[Exception],
    allow_zero: bool = False,
    optional: bool = False,
) -> str:
    """Validate a lowercase 64-hex SHA-256 digest; reject the all-zero digest.

    ``optional=True`` admits the empty string (an explicitly absent digest);
    ``allow_zero=True`` must never be set on identity or receipt fields — the
    all-zero digest is a placeholder, not an identity.
    """

    if optional and value == "":
        return ""
    try:
        digest = require_sha256_hex(value, field=field)
    except ValueError as exc:
        raise error(str(exc)) from None
    if not allow_zero and digest == _ZERO_DIGEST:
        raise error(f"{field} must not be the all-zero digest")
    return digest


def require_int(
    value: object,
    *,
    field: str,
    error: type[Exception],
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Validate an exact ``int`` (bool rejected) within optional bounds."""

    if (
        type(value) is int
        and (minimum is None or value >= minimum)
        and (maximum is None or value <= maximum)
    ):
        return value
    if minimum is not None and maximum is not None:
        raise error(f"{field} must be an integer in [{minimum}, {maximum}]")
    if minimum is not None:
        raise error(f"{field} must be an integer >= {minimum}")
    if maximum is not None:
        raise error(f"{field} must be an integer <= {maximum}")
    raise error(f"{field} must be an integer")


def require_identifier(
    value: object,
    *,
    field: str,
    error: type[Exception],
    pattern: re.Pattern[str],
) -> str:
    """Validate a string against the module's identifier grammar.

    The grammar is deliberately a parameter: arena, settlement, and intake
    identifiers have different alphabets and length caps, and unifying them
    silently would change admission policy.
    """

    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise error(f"{field} is not a canonical identifier")
    return value


def require_exact_fields(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
    error: type[Exception],
    exact_dict: bool = False,
) -> Mapping[str, Any]:
    """Validate an object with string keys and EXACTLY the expected field set.

    ``exact_dict=True`` additionally requires ``type(value) is dict`` (for
    payloads that must come straight from ``json.loads``, never a Mapping
    stand-in).
    """

    if exact_dict:
        if type(value) is not dict:
            raise error(f"{label} must be a JSON object")
    elif not isinstance(value, Mapping):
        raise error(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise error(f"{label} keys must be strings")
    actual = frozenset(value)
    if actual != fields:
        missing = tuple(sorted(fields - actual))
        extra = tuple(sorted(actual - fields))
        raise error(
            f"{label} fields mismatch: missing={missing!r}, extra={extra!r}"
        )
    return value


def duplicate_key_pairs(
    pairs: list[tuple[str, Any]],
    *,
    label: str,
    error: type[Exception],
) -> dict[str, Any]:
    """``json.loads`` object_pairs_hook body that rejects duplicate keys."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise error(f"{label} contains duplicate key {key!r}")
        result[key] = value
    return result


def require_driver_integer(
    value: object,
    *,
    field: str,
    error: type[Exception],
) -> int:
    """Accept integer-protocol CUDA driver values without lossy coercion.

    Driver bindings return plain ints or enum wrappers exposing ``.value``;
    bools and anything non-integral are malformed. ``operator.index`` keeps
    the conversion exact (no ``int()`` truncation).
    """

    if isinstance(value, bool):
        raise error(f"CUDA driver returned a malformed {field}")
    try:
        return operator.index(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        enum_value = getattr(value, "value", None)
        if isinstance(enum_value, bool):
            raise error(f"CUDA driver returned a malformed {field}") from None
        try:
            return operator.index(enum_value)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            raise error(f"CUDA driver returned a malformed {field}") from None


def truthy_flag(value: str | None) -> bool:
    """Parse an environment-flag string ("1"/"true"/"yes"/"on")."""

    return (value or "").strip().lower() in ("1", "true", "yes", "on")
