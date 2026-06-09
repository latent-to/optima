#!/usr/bin/env python3
"""Render a dataset + findings into ONE self-contained HTML dashboard.

No server, no build step, no network: the data is inlined into the page, so the
output ``report.html`` opens by double-click and works fully offline (the Mac
has no local GPU tooling — portability is the whole point).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "templates" / "dashboard.html"
PLACEHOLDER = "__PROFILER_PAYLOAD__"


def render(dataset: dict, findings: dict, out: Path) -> Path:
    html = TEMPLATE.read_text()
    payload = json.dumps({"dataset": dataset, "findings": findings}, ensure_ascii=True)
    # Safe to embed inside a <script type="application/json"> tag.
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    out.write_text(html.replace(PLACEHOLDER, payload))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_json", type=Path)
    ap.add_argument("findings_json", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("report.html"))
    args = ap.parse_args()
    render(json.loads(args.dataset_json.read_text()),
           json.loads(args.findings_json.read_text()), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
