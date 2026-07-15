"""Canonical OCI CPU and NUMA-memory placement policy helpers."""

from __future__ import annotations

import re


_CPUSET = re.compile(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*\Z")
_MAX_CPU_INDEX = 1_048_575
_MAX_MEMORY_NODE_INDEX = 65_535


def _canonical_index_set(
    value: object,
    *,
    field: str,
    maximum: int,
) -> tuple[str | None, int]:
    """Validate Docker/Linux cpuset syntax without expanding attacker-sized ranges.

    ``None`` means that placement is intentionally left to the host.  Present sets
    use one canonical spelling: decimal indices have no leading zeroes, ranges are
    increasing, and adjacent/overlapping ranges are merged.  Requiring that spelling
    makes the policy digest identify exactly one kernel cpuset.
    """

    if value is None:
        return None, 0
    if type(value) is not str or _CPUSET.fullmatch(value) is None:
        raise ValueError(f"{field} must be a canonical cpuset or None")

    previous_end = -2
    cardinality = 0
    for token in value.split(","):
        raw_start, separator, raw_end = token.partition("-")
        if (raw_start != "0" and raw_start.startswith("0")) or (
            raw_end and raw_end != "0" and raw_end.startswith("0")
        ):
            raise ValueError(f"{field} must not contain leading zeroes")
        start = int(raw_start)
        end = int(raw_end) if separator else start
        if end < start:
            raise ValueError(f"{field} ranges must be increasing")
        if separator and end == start:
            raise ValueError(f"{field} singleton ranges must use one index")
        if end > maximum:
            raise ValueError(f"{field} index exceeds its hard bound")
        if start <= previous_end:
            raise ValueError(f"{field} ranges must be sorted and non-overlapping")
        if start == previous_end + 1:
            raise ValueError(f"{field} adjacent ranges must be merged")
        cardinality += end - start + 1
        previous_end = end
    return value, cardinality


def validate_cpuset_pair(
    cpuset_cpus: object,
    cpuset_mems: object,
    *,
    cpu_millis: int,
) -> tuple[str | None, str | None]:
    """Return a canonical paired CPU/memory-node placement policy.

    CPU and memory placement are paired deliberately: accepting just one would make
    an ostensibly isolated NUMA policy silently allocate or execute on another node.
    The CFS quota may be lower than the selected CPU count, but not higher than the
    maximum execution capacity of that set.
    """

    cpus, cpu_count = _canonical_index_set(
        cpuset_cpus,
        field="cpuset_cpus",
        maximum=_MAX_CPU_INDEX,
    )
    mems, _ = _canonical_index_set(
        cpuset_mems,
        field="cpuset_mems",
        maximum=_MAX_MEMORY_NODE_INDEX,
    )
    if (cpus is None) != (mems is None):
        raise ValueError("cpuset_cpus and cpuset_mems must be specified together")
    if cpus is not None and cpu_millis > cpu_count * 1_000:
        raise ValueError("cpu_millis exceeds the selected cpuset_cpus capacity")
    return cpus, mems


__all__ = ["validate_cpuset_pair"]
