#!/usr/bin/env python3
"""One command: ingest a profile dir -> findings -> self-contained HTML report.

    python3 build.py <datadir> [-o outdir]

Writes ``<outdir>/dataset.json``, ``<outdir>/findings.json`` and
``<outdir>/report.html``. Open the HTML in any browser (offline, no deps).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ingest  # noqa: E402
import findings as findings_mod  # noqa: E402
import report  # noqa: E402


def build(datadir: Path, outdir: Path, use_cache: bool = True) -> dict:
    ds = ingest.ingest(datadir, use_cache=use_cache).to_dict()
    fnd = findings_mod.derive(ds)
    ds["findings"] = fnd
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "dataset.json").write_text(json.dumps(ds, indent=2))
    (outdir / "findings.json").write_text(json.dumps(fnd, indent=2))
    report.render(ds, fnd, outdir / "report.html")
    return {"dataset": ds, "findings": fnd, "outdir": outdir}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datadir", type=Path)
    ap.add_argument("-o", "--outdir", type=Path, default=Path("profiler_out"))
    ap.add_argument("--no-cache", action="store_true", help="re-parse torch traces (ignore .profiler_cache)")
    args = ap.parse_args()
    res = build(args.datadir.expanduser(), args.outdir.expanduser(), use_cache=not args.no_cache)
    fnd = res["findings"]
    print(f"== {res['outdir']} ==")
    print(f"report:   {res['outdir']/'report.html'}")
    print(f"dataset:  {res['outdir']/'dataset.json'}")
    print()
    print("HEADLINE:", fnd["headline"])
    print()
    print("Top opportunities:")
    for o in fnd["opportunities"][:6]:
        g = f"+{o['est_decode_gain_pct']}% decode" if o["est_decode_gain_pct"] is not None else "unknown gain"
        print(f"  - {o['title']}  [{g}]")
    print()
    print("Hard constraints:")
    for c in fnd["constraints"]:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
