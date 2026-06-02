"""Extract DECODE-ONLY all-reduce kernel latency from an nsys sqlite — the collective ceiling.

Splits decode (cudagraph-replayed: ``graphId`` set) from prefill (eager: ``graphId`` null/0),
then for every all-reduce / comm kernel reports count, total ms, per-call avg/min/max us, and
% of decode GPU time. The read:

* ``min_us`` ~= the bandwidth floor for these messages (a real, unblocked reduce).
* ``avg_us`` = realized per-call cost. ``avg_us >> min_us`` => the kernel is LATENCY / sync
  bound (exposed wait), not bandwidth bound — quantizing bytes won't help; lower latency or
  compute-comm overlap will. (Measured baseline for the marlin champion: the decode
  ``all_reduce_two_shot`` was avg ~333us vs min ~9us, ~34% of decode.)

Usage: python3 parse_allreduce_latency.py RUN.sqlite  [RUN2.sqlite ...]
"""

from __future__ import annotations

import sqlite3
import sys

_DECODE_TOTAL = (
    "SELECT COALESCE(SUM(end - start), 0) FROM CUPTI_ACTIVITY_KIND_KERNEL "
    "WHERE graphId IS NOT NULL AND graphId != 0"
)

_COMM = """
SELECT CASE WHEN k.graphId IS NULL OR k.graphId = 0 THEN 'prefill' ELSE 'decode' END AS mode,
       substr(s.value, 1, 40) AS name,
       COUNT(*) AS n,
       ROUND(SUM(k.end - k.start) / 1e6, 2) AS tot_ms,
       ROUND(AVG(k.end - k.start) / 1e3, 1) AS avg_us,
       ROUND(MIN(k.end - k.start) / 1e3, 1) AS min_us,
       ROUND(MAX(k.end - k.start) / 1e3, 1) AS max_us
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON s.id = k.demangledName
WHERE s.value LIKE '%llReduce%' OR s.value LIKE '%all_reduce%'
   OR s.value LIKE '%AllGather%' OR s.value LIKE '%nvls%' OR s.value LIKE '%multimem%'
GROUP BY mode, s.value
ORDER BY tot_ms DESC
"""


def report(path: str) -> None:
    con = sqlite3.connect(path)
    try:
        decode_total_ms = con.execute(_DECODE_TOTAL).fetchone()[0] / 1e6
        print(f"\n=== {path} ===")
        print(f"decode GPU time (graph kernels): {decode_total_ms:.1f} ms")
        hdr = ("mode", "kernel", "n", "tot_ms", "avg_us", "min_us", "max_us", "%dec")
        print("%-8s %-40s %6s %9s %8s %8s %9s %6s" % hdr)
        for mode, name, n, tot, avg, mn, mx in con.execute(_COMM):
            pct = 100.0 * tot / decode_total_ms if (mode == "decode" and decode_total_ms > 0) else 0.0
            print("%-8s %-40s %6d %9.2f %8.1f %8.1f %9.1f %6.1f" % (mode, name, n, tot, avg, mn, mx, pct))
    finally:
        con.close()


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: parse_allreduce_latency.py RUN.sqlite [RUN2.sqlite ...]")
        return
    for path in sys.argv[1:]:
        try:
            report(path)
        except Exception as ex:  # noqa: BLE001
            print(f"\n=== {path} ===\nERROR: {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
