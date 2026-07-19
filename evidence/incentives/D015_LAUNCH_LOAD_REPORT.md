# D-015 one-campaign launch-load sensitivity

Status: deterministic replay complete.

Semantic report digest: `505fed4d40a6acc6bc92d6330170e8e2260a52e5f3099c22a6c0eb4b2308c672`

This supplement corrects a narrow assumption in the historical D-015 selection
tape. That tape rotated one aggregate CROWN among target families. This replay
instead lets each of 1, 2, 5, or 10 active MiniMax-M3 target families CROWN
independently every 7, 14, 30, or 90 days. Every claim then receives its full
90-day payment window.

The 14-day arrival window models the announced one-to-two-week launch. The
365-day window is sustained-pressure sensitivity, not a prediction that every
family will repeatedly produce 4.4% wins for a year. All rows use the selected
`k=1`, a 10% reserve, 4.4% relative CROWNs, and either no discovery payouts or a
synthetically saturated 50,000-unit discovery payout in every epoch.

## Launch window: 14 days of arrivals plus the 90-day drain

Collection of issued registered-CROWN principal:

### Empty discovery load

| Independently winning families | Every 7 days | Every 14 days | Every 30 days | Every 90 days |
|---:|---:|---:|---:|---:|
| 1 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 2 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 5 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 10 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |

### Saturated discovery load

| Independently winning families | Every 7 days | Every 14 days | Every 30 days | Every 90 days |
|---:|---:|---:|---:|---:|
| 1 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 2 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 5 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 10 | 99.0211% | 100.0000% | 100.0000% | 100.0000% |

The harshest launch row creates twenty CROWNs—ten on day 0 and ten on day 7—
while discovery consumes its maximum share every day. It pays 99.0211% of
principal; 765,240 of 78,174,980 units expire. Up through five independently
winning weekly families, every tested launch claim clears in full.

## Sustained-pressure control: 365 days of arrivals plus the drain

### Empty discovery load

| Independently winning families | Every 7 days | Every 14 days | Every 30 days | Every 90 days |
|---:|---:|---:|---:|---:|
| 1 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 2 | 89.6771% | 100.0000% | 100.0000% | 100.0000% |
| 5 | 39.1861% | 73.1470% | 100.0000% | 100.0000% |
| 10 | 19.6555% | 38.3590% | 75.1740% | 100.0000% |

### Saturated discovery load

| Independently winning families | Every 7 days | Every 14 days | Every 30 days | Every 90 days |
|---:|---:|---:|---:|---:|
| 1 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 2 | 85.6341% | 100.0000% | 100.0000% | 100.0000% |
| 5 | 37.1198% | 69.5918% | 100.0000% | 100.0000% |
| 10 | 18.5635% | 36.2279% | 71.5328% | 100.0000% |

This is the overload warning the rotating-family tape could not expose. The
finite 90-day claim remains solvent by construction—unpaid units expire—but a
miner cannot assume near-100% collection if many families keep producing large
wins indefinitely.

## Rental sensitivity for the launch

A first 4.4% one-campaign CROWN issues 3,894,697 units. At a hypothetical 25%
chance of winning and a $1,000–$1,500 rental campaign:

| 14-day weekly load, saturated discovery | Measured collection | Break-even full-vector-day value |
|---|---:|---:|
| 1 independently winning family | 100.0000% | $1,027.04–$1,540.56 |
| 2 independently winning families | 100.0000% | $1,027.04–$1,540.56 |
| 5 independently winning families | 100.0000% | $1,027.04–$1,540.56 |
| 10 independently winning families | 99.0211% | $1,037.20–$1,555.79 |

These values are algebraic sensitivities, not token-price or payout promises.
The report also records an ROI row against the measured collection fraction of
every matrix cell. Expanding to a second model campaign is deliberately deferred;
it requires a successor policy and a new load study.

## Burst boundary

A same-day burst of nineteen first 4.4% CROWNs pays fully with saturated
discovery. Twenty pays 98.2104%, with 1,393,940 units expiring. With no discovery
load, both bursts pay fully.

## Reproduction

The replay uses production claim issuance and the production digest-ordered
integer pro-rata rule. It requires no GPU, wallet, chain access, or private
experiment ledger:

```bash
python -m scripts.d015_launch_load \
  --config evidence/incentives/d015_launch_load_config.json \
  --out /tmp/d015_launch_load_report.json
```

The tracked CI test regenerates all 64 matrix cells and four burst controls,
checks the semantic digest above, and separately proves that campaign shares
size claim principal rather than creating hard payout silos.

Boundaries: this is deterministic accounting sensitivity for one model campaign.
It is not evidence about actual win frequency, miner equilibrium, token value,
GPU performance, or a future two-campaign configuration. Aggregate tape
collection is not a guarantee that any particular miner collects that fraction.
