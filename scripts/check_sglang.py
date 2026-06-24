#!/usr/bin/env python3
"""Weekly sglang canary: is there a newer sglang, and do our seams still hold?

Two checks, both cheap:
  1. PyPI release check (pure HTTP, no deps) — is there a newer sglang than the
     pinned one? This is the "should we consider a bump?" signal.
  2. The seam compat canary (`optima.compat`) — if sglang is importable here, do
     our integration points still exist? (Full behavioral confirmation needs a GPU
     box; see docs/SGLANG_TRACKING.md.)

Exit 1 ONLY on a broken seam (a chokepoint we patch moved — code-actionable). A newer
sglang release is a WARNING (GitHub annotation), not a failure: bumping is a deliberate
human decision (re-baseline), not a code fix, so it must not turn the weekly canary red.

Schedule it: the GitHub Action in .github/workflows/sglang-canary.yml, or cron:
    0 9 * * 1  cd /path/to/optima && .venv/bin/python scripts/check_sglang.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_ADDED_ROOT = False
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    _ADDED_ROOT = True


def latest_sglang() -> str | None:
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/sglang/json", timeout=20) as r:
            return json.load(r)["info"]["version"]
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not reach PyPI: {exc})")
        return None


def main() -> int:
    try:
        from optima.compat import PINNED_SGLANG, format_checks, run_checks
    except Exception as exc:  # noqa: BLE001
        print(f"cannot import optima.compat (install the harness: pip install -e .): {exc}")
        return 1
    finally:
        # Avoid shadowing an installed sglang package with the repo's vendored
        # `sglang/` source tree when this script is run from a checkout.
        if _ADDED_ROOT:
            try:
                sys.path.remove(str(ROOT))
            except ValueError:
                pass

    # Two SEPARATE signals — only one is code-actionable, so only one fails CI:
    #   * seam_broken  -> a chokepoint we patch moved/changed. Fix the adapter. HARD FAIL (exit 1).
    #   * newer_release -> sglang shipped a version past the pin. A human bump+re-baseline decision
    #                      (docs/SGLANG_TRACKING.md), NOT a code fix. WARNING only (does not fail CI),
    #                      else the canary goes red on every sglang release = alert fatigue.
    seam_broken = False
    newer_release = False
    print("=== sglang canary ===")
    print(f"pinned (scored version): {PINNED_SGLANG}")

    latest = latest_sglang()
    if latest:
        print(f"latest on PyPI:          {latest}")
        if latest != PINNED_SGLANG:
            msg = (f"newer sglang available: {PINNED_SGLANG} -> {latest}. "
                   "Consider the bump+re-baseline process (docs/SGLANG_TRACKING.md).")
            print(f"  -> {msg}")
            print(f"::warning title=sglang newer release::{msg}")  # GitHub annotation, non-failing
            newer_release = True
        else:
            print("  -> up to date.")

    try:
        import sglang  # noqa: F401
    except Exception:  # noqa: BLE001
        print("\n(sglang not importable here — can't run the seam canary; "
              "run `optima compat` on a pod/venv with sglang installed)")
        # No way to verify seams here -> nothing code-actionable -> don't fail CI.
        return 0

    print("\nseam compat canary (installed sglang):")
    checks = run_checks()
    print(format_checks(checks))
    if not all(c.ok for c in checks):
        broken = ", ".join(c.name for c in checks if not c.ok)
        print(f"::error title=sglang seam broken::chokepoint(s) moved: {broken}. "
              "Fix the seam adapter (optima/integrations/, see optima/seams.py).")
        seam_broken = True

    if newer_release and not seam_broken:
        print("\nresult: seams INTACT; a newer release exists (informational, not a failure).")
    return 1 if seam_broken else 0


if __name__ == "__main__":
    sys.exit(main())
