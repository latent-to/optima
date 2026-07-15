"""Canonical entry point for the isolated CuTe AOT compiler child.

Running ``optima.cute_aot`` itself with ``python -m`` would execute that module as
``__main__``.  Candidate factories correctly import ``optima.cute_aot`` and would
then see a second copy of its request class.  This tiny validator-owned entry point
keeps the implementation under its canonical module identity, so exact request-type
validation remains meaningful.
"""

from __future__ import annotations

from optima import cute_aot


def main(argv: list[str] | None = None) -> int:
    return cute_aot._main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
