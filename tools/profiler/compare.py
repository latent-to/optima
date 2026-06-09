#!/usr/bin/env python3
"""A/B two profile datasets → one win/no-win verdict.

The honest question for a kernel patch is *baseline vs patched*, and this turns
"did my kernel win?" from eyeballing two reports into one verdict line.

    python3 compare.py base/dataset.json patched/dataset.json [-o compare.html]

Arbitration (deliberately strict — see why below):
  * **E2E throughput is the arbiter.** It's steady-state and directly comparable
    across runs; kernel-time shares are not (the denominator moves when you fuse).
  * **A launch-count drop is the fusion proof.** Fusing eliminates a kernel
    boundary, so the glue category's launches collapse — a presence/absence
    signal that's robust to trace-window length.
  * **Within ±noise is NOT a win.** A throughput delta below the noise floor
    (clock/thermal — the scorer-margin trap) is INCONCLUSIVE, not a win. Re-run
    interleaved + clock-locked to confirm. Default noise 2%% (e2e is far steadier
    than isolated-kernel timing); raise it with --noise-pct.

So a *real* win reads: e2e up past noise AND a glue category's launches dropped.
e2e up with no structural change → suspect clock noise. e2e flat → no win.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_NOISE_PCT = 2.0
FUSION_DROP = 0.5   # a winnable category whose launches fall below this fraction = fused


def _peak(findings: dict) -> dict | None:
    return findings.get("peak")


def _decode_cats(findings: dict) -> dict[str, dict]:
    cats = (findings.get("decode_canonical") or {}).get("categories") or []
    return {c["cat"]: c for c in cats}


def _e2e_primary(dataset: dict) -> dict[tuple[str, int], float]:
    """matched-able e2e points: (config, conc) -> agg tok/s, primary configs only."""
    out = {}
    for r in dataset.get("e2e", []):
        if r.get("config") in ("mtp_off", "mtp_off_r2", "mtp_on"):
            out[(r["config"], r["conc"])] = r["agg_toks"]
    return out


def compare(a: dict, b: dict, label_a="baseline", label_b="patched",
            noise_pct=DEFAULT_NOISE_PCT) -> dict:
    fa, fb = a.get("findings", {}), b.get("findings", {})

    # ---- e2e: peak + matched per-(config,conc) ----
    pa, pb = _peak(fa), _peak(fb)
    peak_delta = None
    if pa and pb and pa["tok_s"]:
        peak_delta = round(100 * (pb["tok_s"] - pa["tok_s"]) / pa["tok_s"], 1)
    ea, eb = _e2e_primary(a), _e2e_primary(b)
    e2e_rows = []
    for key in sorted(set(ea) & set(eb)):
        va, vb = ea[key], eb[key]
        e2e_rows.append({"config": key[0], "conc": key[1], "a": va, "b": vb,
                         "delta_pct": round(100 * (vb - va) / va, 1) if va else None})

    # ---- decode structure: per-category share/us/launches ----
    ca, cb = _decode_cats(fa), _decode_cats(fb)
    display = a.get("display", {})
    decode_rows = []
    for cat in sorted(set(ca) | set(cb), key=lambda k: -(ca.get(k, cb.get(k, {})).get("pct", 0))):
        ra, rb = ca.get(cat), cb.get(cat)
        a_count = ra["count"] if ra else 0
        b_count = rb["count"] if rb else 0
        decode_rows.append({
            "cat": cat, "display": display.get(cat, cat),
            "a_pct": ra["pct"] if ra else 0.0, "b_pct": rb["pct"] if rb else 0.0,
            "a_count": a_count, "b_count": b_count,
            "count_delta": b_count - a_count,
            "count_ratio": (b_count / a_count) if a_count else None,
            "a_bound": ra["bound_type"] if ra else "—", "b_bound": rb["bound_type"] if rb else "—",
            "a_winnable": ra.get("winnable") if ra else None,
        })

    # ---- fusion evidence: winnable (glue) categories whose launches collapsed ----
    fused = [r for r in decode_rows
             if r["a_winnable"] is True and r["count_ratio"] is not None and r["count_ratio"] < FUSION_DROP]

    # ---- verdict ----
    signals, cautions = [], []
    win = None
    delta = peak_delta
    if delta is None:
        headline = "NO E2E DATA — cannot arbitrate (need serve_load2 results in both datasets)."
    elif delta > noise_pct:
        if fused:
            win = True
            names = ", ".join(f"{f['display']} ({f['a_count']}→{f['b_count']} launches)" for f in fused)
            headline = f"WIN: +{delta}% e2e peak, corroborated by fusion of {names}."
            signals.append(f"e2e peak {pa['tok_s']:.0f} → {pb['tok_s']:.0f} tok/s (+{delta}%)")
            signals.append(f"launch boundaries eliminated: {names}")
        else:
            win = True
            headline = (f"APPARENT WIN: +{delta}% e2e peak — but NO kernel-structure change "
                        f"(no glue category's launches dropped). Confirm it isn't clock/thermal "
                        f"noise: re-run interleaved + clock-locked before trusting it.")
            cautions.append("e2e rose but no fusion signal — a structural win should also drop a launch count")
    elif delta < -noise_pct:
        win = False
        headline = f"REGRESSION: {delta}% e2e peak. The patch made it slower."
    else:
        win = None
        headline = (f"INCONCLUSIVE: e2e peak Δ {delta:+}% is within ±{noise_pct}% noise — NOT a win. "
                    f"If you expect a win, re-run interleaved + clock-locked to resolve it.")

    # bound-type changes worth noting
    for r in decode_rows:
        if r["a_bound"] != r["b_bound"] and r["a_bound"] != "—" and r["b_bound"] != "—":
            cautions.append(f"{r['display']}: bound-type {r['a_bound']} → {r['b_bound']}")

    return {
        "label_a": label_a, "label_b": label_b, "noise_pct": noise_pct,
        "win": win, "headline": headline, "signals": signals, "cautions": cautions,
        "peak": {"a": pa, "b": pb, "delta_pct": peak_delta},
        "e2e": e2e_rows, "decode": decode_rows, "fused": fused,
        "meta": {"a": a.get("meta", {}).get("datadir"), "b": b.get("meta", {}).get("datadir")},
    }


# --------------------------------------------------------------------------- #
# renderers
# --------------------------------------------------------------------------- #
def render_text(c: dict) -> str:
    L = [f"\n=== {c['label_a']}  →  {c['label_b']} ===", "", c["headline"], ""]
    if c["signals"]:
        L.append("signals:")
        L += [f"  ✓ {s}" for s in c["signals"]]
    if c["cautions"]:
        L.append("cautions:")
        L += [f"  ! {s}" for s in c["cautions"]]
    L.append("")
    L.append(f"{'e2e (config/conc)':24} {c['label_a']:>11} {c['label_b']:>11} {'Δ%':>8}")
    for r in c["e2e"]:
        L.append(f"  {r['config']+' c'+str(r['conc']):22} {r['a']:11.0f} {r['b']:11.0f} {r['delta_pct']:+8.1f}")
    L.append("")
    L.append(f"{'decode category':28} {'share% a→b':>14} {'launches a→b':>16} {'bound a→b':>20}")
    for r in c["decode"]:
        if r["a_pct"] < 0.3 and r["b_pct"] < 0.3:
            continue
        sh = f"{r['a_pct']:.1f}→{r['b_pct']:.1f}"
        lc = f"{r['a_count']}→{r['b_count']}"
        bd = f"{r['a_bound']}→{r['b_bound']}"
        flag = "  ⟵ FUSED" if r in c["fused"] else ""
        L.append(f"  {r['display'][:26]:28} {sh:>14} {lc:>16} {bd:>20}{flag}")
    return "\n".join(L)


def render_html(c: dict) -> str:
    win = c["win"]
    color = "#3fb950" if win is True else "#f85149" if win is False else "#d29922"

    def dcell(v, good_when_positive=True):
        if v is None:
            return '<td class="num">—</td>'
        col = "#3fb950" if (v > 0) == good_when_positive and v != 0 else "#f85149" if v != 0 else "#8b949e"
        return f'<td class="num" style="color:{col}">{v:+.1f}</td>'

    e2e = "".join(
        f"<tr><td>{r['config']} c{r['conc']}</td><td class=num>{r['a']:.0f}</td>"
        f"<td class=num>{r['b']:.0f}</td>{dcell(r['delta_pct'])}</tr>" for r in c["e2e"])
    dec = ""
    for r in c["decode"]:
        if r["a_pct"] < 0.3 and r["b_pct"] < 0.3:
            continue
        fused = r in c["fused"]
        cd = r["count_delta"]
        cdcol = "#3fb950" if cd < 0 else "#f85149" if cd > 0 else "#8b949e"
        dec += (f"<tr style='{'background:#11251a' if fused else ''}'>"
                f"<td>{r['display']}{' <b style=color:#3fb950>⟵ FUSED</b>' if fused else ''}</td>"
                f"<td class=num>{r['a_pct']:.1f} → {r['b_pct']:.1f}</td>"
                f"<td class=num>{r['a_count']} → {r['b_count']} "
                f"<span style='color:{cdcol}'>({cd:+})</span></td>"
                f"<td class=num>{r['a_bound']} → {r['b_bound']}</td></tr>")
    sig = "".join(f"<li>✓ {s}</li>" for s in c["signals"])
    cau = "".join(f"<li>! {s}</li>" for s in c["cautions"])
    return f"""<!DOCTYPE html><html><head><meta charset=utf-8><title>A/B · {c['label_a']} vs {c['label_b']}</title>
<style>
 body{{margin:0;background:#0d1117;color:#e6edf3;font:14px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}}
 header{{padding:22px 26px;border-bottom:1px solid #30363d}} h1{{font-size:18px;margin:0 0 10px}}
 .verdict{{border:1px solid #30363d;border-left:4px solid {color};border-radius:8px;padding:12px 14px;background:#161b22;font-size:15px}}
 main{{padding:18px 26px;max-width:1000px}} h2{{font-size:15px;border-bottom:1px solid #30363d;padding-bottom:6px;margin:26px 0 12px}}
 table{{width:100%;border-collapse:collapse;font-size:13px}} td,th{{text-align:left;padding:6px 10px;border-bottom:1px solid #30363d}}
 td.num,th.num{{text-align:right;font-family:ui-monospace,Menlo,monospace}} th{{color:#8b949e}}
 ul{{color:#8b949e;font-size:13px}} .lab{{color:#8b949e;font-size:12px}}
</style></head><body>
<header><h1>A/B comparison &middot; <span class=lab>{c['label_a']}</span> &rarr; <span class=lab>{c['label_b']}</span></h1>
<div class=verdict>{c['headline']}</div></header>
<main>
{'<h2>Signals</h2><ul>'+sig+'</ul>' if sig else ''}
{'<h2>Cautions</h2><ul>'+cau+'</ul>' if cau else ''}
<h2>E2E throughput (the arbiter)</h2>
<table><tr><th>config</th><th class=num>{c['label_a']}</th><th class=num>{c['label_b']}</th><th class=num>Δ%</th></tr>{e2e}</table>
<h2>Decode structure (corroboration — launch-count drop = fusion proof)</h2>
<table><tr><th>category</th><th class=num>share% a→b</th><th class=num>launches a→b</th><th class=num>bound a→b</th></tr>{dec}</table>
<p class=lab>share% is normalized (denominator shifts when you fuse) — corroboration, not arbitration. E2E is the arbiter.</p>
</main></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path, help="baseline dataset.json")
    ap.add_argument("patched", type=Path, help="patched dataset.json")
    ap.add_argument("--label-a", default="baseline")
    ap.add_argument("--label-b", default="patched")
    ap.add_argument("--noise-pct", type=float, default=DEFAULT_NOISE_PCT)
    ap.add_argument("-o", "--out", type=Path, default=None, help="write compare.html here")
    args = ap.parse_args()
    a = json.loads(args.base.read_text())
    b = json.loads(args.patched.read_text())
    c = compare(a, b, args.label_a, args.label_b, args.noise_pct)
    print(render_text(c))
    if args.out:
        args.out.write_text(render_html(c))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
