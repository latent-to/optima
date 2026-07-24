# One-campaign incentive load validation

This deterministic replay measures how the selected one-campaign finite-debt
arithmetic behaves when several MiniMax-M3 reward families produce independent
4.4% crowns. It is accounting sensitivity, not a forecast of win frequency,
token price, miner equilibrium, validator influence, or GPU performance.

Status: deterministic replay complete.

Semantic report digest: `505fed4d40a6acc6bc92d6330170e8e2260a52e5f3099c22a6c0eb4b2308c672`

## Model

The replay varies:

- `1`, `2`, `5`, or `10` independently winning reward families;
- one 4.4% crown per active family every `7`, `14`, `30`, or `90` days;
- either no discovery payout or a synthetic `50,000`-unit discovery payout in
  every epoch; and
- a 14-day arrival window or a 365-day sustained-pressure window.

Every issued claim then receives its complete 90-day payment window. All rows
use the selected `k = 1`, a 10% reserve, a single campaign with 100% claim
sizing, and the production claim-digest integer pro-rata rule.

The 14-day arrival window is a bounded launch sensitivity. The 365-day window
is a pressure control. Neither asserts that every family will repeatedly
produce a 4.4% crown at the modeled cadence.

## Fourteen-day arrival window

The following tables report the fraction of issued registered-CROWN principal
paid after the final 90-day drain.

### No discovery payout

| Independently winning families | Every 7 days | Every 14 days | Every 30 days | Every 90 days |
|---:|---:|---:|---:|---:|
| 1 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 2 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 5 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 10 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |

### Saturated discovery payout

| Independently winning families | Every 7 days | Every 14 days | Every 30 days | Every 90 days |
|---:|---:|---:|---:|---:|
| 1 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 2 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 5 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 10 | 99.0211% | 100.0000% | 100.0000% | 100.0000% |

The highest-load launch row creates twenty claims: ten at day 0 and ten at day
7, while discovery consumes its full capacity in every epoch. It pays 99.0211%
of principal; `765,240` of `78,174,980` units expire. Every tested launch row
through five independently winning weekly families pays in full.

## Sustained-pressure control

### No discovery payout

| Independently winning families | Every 7 days | Every 14 days | Every 30 days | Every 90 days |
|---:|---:|---:|---:|---:|
| 1 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 2 | 89.6771% | 100.0000% | 100.0000% | 100.0000% |
| 5 | 39.1861% | 73.1470% | 100.0000% | 100.0000% |
| 10 | 19.6555% | 38.3590% | 75.1740% | 100.0000% |

### Saturated discovery payout

| Independently winning families | Every 7 days | Every 14 days | Every 30 days | Every 90 days |
|---:|---:|---:|---:|---:|
| 1 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 2 | 85.6341% | 100.0000% | 100.0000% | 100.0000% |
| 5 | 37.1198% | 69.5918% | 100.0000% | 100.0000% |
| 10 | 18.5635% | 36.2279% | 71.5328% | 100.0000% |

Finite expiry bounds liability, but it does not guarantee full collection under
sustained concurrent issuance. In the saturated weekly rows, one family
collects all principal, while two, five, and ten families collect 85.6341%,
37.1198%, and 18.5635% respectively.

## Same-day burst boundary

A same-day burst of nineteen first 4.4% crowns pays fully with saturated
discovery. Twenty pays 98.2104%, with `1,393,940` units expiring. With no
discovery payout, both burst controls pay fully.

## Rental-cost sensitivity

A first 4.4% one-campaign crown issues `3,894,697` units. For a hypothetical
25% success probability and a `$1,000-$1,500` optimization campaign:

| 14-day weekly load with saturated discovery | Measured collection | Break-even full-vector-day value |
|---|---:|---:|
| 1 independently winning family | 100.0000% | $1,027.04-$1,540.56 |
| 2 independently winning families | 100.0000% | $1,027.04-$1,540.56 |
| 5 independently winning families | 100.0000% | $1,027.04-$1,540.56 |
| 10 independently winning families | 99.0211% | $1,037.20-$1,555.79 |

These are algebraic break-even sensitivities, not token-price or payout
promises. The report records an ROI row against every measured matrix cell.

## Reproduction

The tracked configuration is
[`tests/fixtures/incentives/d015_launch_load_config.json`](https://github.com/latent-to/cacheon/blob/main/tests/fixtures/incentives/d015_launch_load_config.json).
The replay uses production issuance from
[`optima/finite_debt.py`](https://github.com/latent-to/cacheon/blob/main/optima/finite_debt.py)
and composition from
[`optima/incentive_composition.py`](https://github.com/latent-to/cacheon/blob/main/optima/incentive_composition.py).

It requires no GPU, wallet, chain access, or private experiment ledger:

```bash
python -m scripts.d015_launch_load \
  --config tests/fixtures/incentives/d015_launch_load_config.json \
  --out /tmp/d015_launch_load_report.json
```

The tracked test regenerates all 64 matrix cells and four burst controls,
verifies the semantic digest, and proves separately that campaign shares size
claim principal rather than creating hard payout silos:

```bash
pytest -q tests/test_d015_launch_load.py
```

## Interpretation boundary

The collection fraction is aggregate across each synthetic tape. It does not
guarantee that a particular miner or claim collects that fraction. A second
simultaneous model campaign is outside the selected launch policy and requires
a successor protocol and a new load study.

For the implemented policy and activation status, see
[Incentives](../miner-guide/incentives.md).
