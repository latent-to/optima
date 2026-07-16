from __future__ import annotations

import sys
from types import ModuleType

from optima.compat import PINNED_SGLANG, run_checks


def _version_check(monkeypatch, version: str):
    sglang = ModuleType("sglang")
    sglang.__version__ = version
    monkeypatch.setitem(sys.modules, "sglang", sglang)

    checks = run_checks()

    return next(
        row for row in checks if row.name == f"sglang installed (pinned {PINNED_SGLANG})"
    )


def test_compat_accepts_the_exact_canonical_sglang_pin(monkeypatch) -> None:
    row = _version_check(monkeypatch, PINNED_SGLANG)

    assert row.ok
    assert row.detail == f"found {PINNED_SGLANG}"


def test_compat_rejects_an_installed_sglang_version_outside_the_pin(monkeypatch) -> None:
    version = "0.0.0.dev1+g56e290315"

    row = _version_check(monkeypatch, version)

    assert not row.ok
    assert row.detail == f"found {version}  <-- DIFFERS from pin"
