# How a real subnet is built — lessons from Affine, applied to Optima

I studied [AffineFoundation/affine-cortex](https://github.com/AffineFoundation/affine-cortex)
(a live Bittensor subnet) to learn the *scaffolding* a production subnet needs —
the chain plumbing, the service decomposition, the state/DB layer, deployment —
independent of its specific incentive (Affine evaluates submitted **models** on RL
tasks; Optima evaluates submitted **kernels** for throughput).

The durable lesson from Affine is its **operational scaffolding**, not its
winner-take-all economics. This document began before Optima's chain and incentive
work matured; the incentive comparison below is retained as historical research.
The current authority is [EMISSIONS_POLICY.md](EMISSIONS_POLICY.md), with the
miner-facing selected curve in [INCENTIVES.md](INCENTIVES.md).

---

## 1. Historical comparison (incentive column superseded)

Side by side:

| Concept | Affine | Optima (what we built) |
|---|---|---|
| Miners submit | `(HF model, revision)` committed on chain | a kernel bundle (commit-reveal) |
| Validator runs the compute | yes (Targon / B300 fleet hosts inference) | yes (our validator runs the model) |
| Champion / challenger | yes | yes (transactional target settlement, `optima/settlement.py`) |
| Dethrone rule | win **strictly across all envs** by a per-env **margin** | beat champion by **speedup margin** |
| Reward | **winner-take-all** to champion (+ burn) | **Superseded:** legacy V1 is the only publisher exercised live. Selected V2 implements finite log-relative CROWN debt plus bounded reviewed discovery, but remains inactive and rejects promotion. |
| Anti-copy | behavioral fingerprint, earliest-committer-wins | content-hash, earliest-commit-wins |
| Anti-overfit | refresh task pool every ~24h, re-sample champion | per-epoch prompt sampling |
| Fairness | judge both on the **same task_ids** in the same window | judge both on the **same prompts** |

The comparison validated the champion/challenger and service architecture. It did
not validate winner-take-all rewards for Optima. Affine adds four operational
refinements worth adopting (§7).

---

## 2. The thing we were missing: a subnet is a *fleet of services around a DB*

Our harness is one CLI. A production subnet is **6-ish cooperating services that
coordinate through a database**, with the chain used only as the trust anchor.
Affine's decomposition (`affine/src/*`):

| Service | Job | Touches chain? |
|---|---|---|
| **validator** | the *only* thing that sets weights on chain; reads computed weights from the DB/API and emits them every ~180 blocks | **yes — weights** |
| **monitor** | syncs the metagraph + reads on-chain miner commitments; seeds the challenger queue; flips `is_valid` | **yes — reads** |
| **scheduler** | the king-of-the-hill engine: block-tick loop that picks a challenger, runs the battle, decides, updates the champion | no |
| **executor** | runs the actual evaluations (process pool, one per env), writes results to the DB | no |
| **anticopy** (refresh + worker) | builds the champion's rollout fingerprints, teacher-forces challengers, computes copy verdicts | no |
| **api** | read-only FastAPI (rank / scores / miners) — the dashboard backend | no |

Key architectural principle: **no message brokers.** Everything coordinates
through the DB. State is *implicit* — the scheduler/executor infer what to do from
"what rows are missing," so the system is restart-safe (re-derive, don't replay).

### Optima's production architecture (the target)

```
            ┌──────── Bittensor chain ────────┐
            │  miner commitments  │  weights   │
            └─────────▲───────────┴─────▲──────┘
                      │ read              │ write
   ┌──────────┐   ┌───┴─────┐        ┌────┴──────┐
   │ monitor  │──►│   DB    │◄───────│ validator │   (thin; sets weights/180 blk)
   │ (chain   │   │ (state) │        └───────────┘
   │  sync,   │   └──▲───┬──┘
   │  queue)  │      │   │
   └──────────┘   ┌──┴───▼────┐   ┌────────────┐   ┌─────────┐
                  │ scheduler │──►│  executor  │──►│ GPU box │  (runs miner kernels
                  │ (king of  │   │ (evaluate: │   │ sglang  │   in isolation)
                  │  the hill)│   │  thru+KL)  │   └─────────┘
                  └───────────┘   └────────────┘
                        │
                  ┌─────▼─────┐
                  │    api    │  (leaderboard / dashboard)
                  └───────────┘
```

Our current code maps cleanly onto this: `chain/intake.py` (the SQLite authority)
→ the **DB + the settlement part of the scheduler**; the qualification runner →
the **executor**; `cli.py` → the seams of all of them. What was new at the time of
this study was the **chain plumbing (monitor + validator)**, a **real DB**, and the
**service split** — all since built (`optima/chain/`).

---

## 3. Chain plumbing — the concrete Bittensor calls

Affine's pattern (`affine/utils/subtensor.py`, `affine/core/miners.py`,
`affine/src/validator/weight_setter.py`). Verify exact signatures against your
`bittensor` version, but the shape is:

```python
import bittensor as bt

# connect (async SDK), with a primary + fallback endpoint
st = bt.AsyncSubtensor("finney")          # or a wss:// URL
await st.initialize()

# metagraph: UID <-> hotkey
meta = await st.metagraph(netuid)
hotkey = meta.hotkeys[uid]

# read miner commitments  (the bundle reference miners put on chain)
commits = await st.get_all_revealed_commitments(netuid)
#   -> { hotkey: [(block, json_str), ...] }
block, payload = commits[hotkey][-1]
ref = json.loads(payload)                 # e.g. {"bundle_hash": ..., "url": ...}

# block cadence for weight setting (epoch-aligned)
blk = await st.get_current_block()
await st.wait_for_block(blk + 1)

# Affine example: set winner-take-all weights to its champion.
# Optima does not use this reward rule; see docs/EMISSIONS_POLICY.md.
await st.set_weights(wallet=wallet, netuid=netuid,
                     uids=uids, weights=weights,
                     wait_for_inclusion=True, wait_for_finalization=True)

# wallet
wallet = bt.Wallet(name=BT_WALLET_COLD, hotkey=BT_WALLET_HOT)
```

Miner side: commit on chain with `set_reveal_commitment(wallet, netuid,
data=json_str, blocks_until_reveal=1)`.

### The big realization: **Bittensor gives you commit-reveal for free**

We had built a local commit-reveal simulator from scratch (since deleted). We
don't need the *transport* — the chain has native commit-reveal
(`set_reveal_commitment` / `get_all_revealed_commitments`). So the design
simplifies:

- **Miner** commits a small JSON on chain: the **bundle hash + a URL** to fetch the
  bundle from (content-addressed store / R2 / HF). The chain timestamps it
  (`first_block`) — that's our anti-copy priority for free.
- **Validator** reads commitments, fetches bundles, evaluates.
- Off-chain we keep only copy detection (`copy_fingerprint.py` + intake's copy
  disposition) and settlement. The commit/reveal *binding* moves to the chain.

Affine uses **plain `set_weights`** (no commit-reveal weights, no `version_key`),
emitted every ~180 blocks, epoch-aligned. Winner gets 1.0, a configurable **burn
fraction** goes to UID 0.

---

## 4. State lives in a DB, not on the chain

The chain holds only **commitments** (miner → bundle ref) and **weights**.
Everything else — scores, the champion, per-miner status, snapshots, the queue —
lives in a database. Affine uses **DynamoDB**; the table shapes generalize:

| Table | Holds |
|---|---|
| `miners` | uid, hotkey, bundle ref, `is_valid`, `challenge_status` |
| `sample_results` | raw per-eval outputs, keyed by `(miner, env, task_id, refresh_block)` |
| `scores` | per-window aggregate score per miner |
| `score_snapshots` | one row per settle: block, outcome, final weights (audit) |
| `system_config` | operator-tunable settings + runtime state (champion, task pool) |

For Optima, our JSON `Ledger` is the toy version of `miners` + `scores` +
`system_config`. Production swaps it for Postgres/DynamoDB with the same shapes.
**Single-writer pattern**: exactly one service writes the scores/weights tables
(Affine's `weight_writer`); everyone else reads. Prevents races.

---

## 5. Miners submit a *reference*, the validator runs the compute

Affine miners never run inference hardware — they commit `(model, revision)` and
the **validator hosts the inference** (Targon, a GPU broker, or an
operator-managed B300 fleet). This is exactly Optima's principle ("miners don't
submit hardware"). **Your 8×B200 ask is literally Affine's "operator-managed
fleet."** Affine even runs sglang on the GPU boxes — same stack as us.

Provider abstraction is worth copying: the scheduler dispatches evals to either a
broker (Targon) or your own fleet (SSH to bare-metal), behind one interface
(`inference_endpoints` table). Start on a broker/rented box, move to owned B200s
when funded — no code change.

---

## 6. Copy detection: exact hashes aren't enough — fingerprint *behavior*

This is the most important technical lesson for us. Our copy detection is an
**exact content hash** — perfect for byte-identical resubmissions. But Affine
fingerprints **behavior**, because a copy is rarely byte-identical:

- For models, Affine compares **sparse logprobs on "decision positions"** — only
  the tokens where the model was *uncertain* (reference logprob below a cutoff).
  Trivial tokens (everyone agrees, lp≈0) are excluded; divergence shows on the
  hard tokens. Copy iff the **median |Δlogp|** across those positions is below a
  threshold (0.05). Plus a **tokenizer signature** (SHA256 over vocab+merges) to
  group comparable models, and a **lookback window** (7d) to avoid cross-season
  false positives. Earliest committer wins.

**Why this matters for Optima:** a kernel "copy" that renames variables, reorders
lines, or tweaks a constant has a **different content hash but identical
behavior**. Exact-hash copy detection misses it. We need a **functional
fingerprint** alongside the source hash — e.g., hash the kernel's **outputs on a
fixed canonical input set** (and/or a normalized-AST hash). Two kernels with the
same output fingerprint within tolerance, where one committed earlier, → the later
is a copy. The normalized-AST half of this has since been built
(`copy_fingerprint.py`); the behavioral half remains open.

(Affine's behavioral approach is heavy — separate worker, R2 storage, async
verdict backfill — because their submissions are giant models. Ours is lighter: a
kernel's output fingerprint is cheap to compute during the eval we already run.)

---

## 7. Four mechanism refinements to adopt from Affine

1. **Re-sample the champion when the task pool refreshes.** When prompts rotate,
   the champion's old score is on old prompts and isn't comparable to a challenger
   scored on new prompts. Affine re-scores the champion on the new pool before the
   queue continues. We must too (today our champion score is frozen at crowning).
2. **Permanently terminate a losing submission.** Affine never re-evaluates a
   challenger that lost (it's terminated; the queue advances). For us: terminate
   per **content/output hash** — a miner can submit a *new, improved* kernel (new
   hash), but the same losing kernel is never re-run. Bounds eval cost and spam.
3. **Overlap-based fairness + oversampling.** Judge champion and challenger on the
   **exact same** task set in the same window; oversample ~10% and abandon the long
   tail so the comparison set is always fully covered.
4. **Burn / reserve fraction.** Affine's configurable UID-0 burn is an emission-control
   lever. Optima's selected analogue is an explicit policy-bound reserve hotkey with a
   100,000-ppm floor: composed payout is `P_d=min(50,000, discovery debt)`, then
   `P_c=min(900,000-P_d, CROWN debt)`, with the remainder to reserve. This is not
   implicitly UID 0, and the production reserve identity is still an activation input.

---

## 8. The security pattern that solves our isolation problem

Affine's anticopy worker uses an **SSH-tunnel isolation** pattern that is exactly
what Optima needs for running untrusted kernels:

> Secrets (chain keys, cloud/R2 credentials) live **only on a CPU control box**.
> The **GPU box** runs only sglang and **never sees any secrets**. The CPU box
> drives the GPU box over an SSH tunnel.

Map this onto our threat model (HOW_OPTIMA_WORKS Part 8.4): the GPU box is where
the **untrusted miner kernel executes**. If that box holds no chain keys, no cloud
creds, and has **no network egress** except the SSH control channel, then even a
fully malicious kernel that achieves code execution:

- can't steal the validator's chain/cloud keys (they're not there),
- can't exfiltrate over the network (no egress),
- can be wiped between evals (ephemeral GPU box).

Combine with a per-eval CUDA context + watchdog (for GPU DoS / OOB writes) and you
have a real isolation boundary. **This is the concrete shape of the isolation
layer we said we still need to build** — and Affine is already running it.

---

## 9. Operational patterns worth stealing

- **Primary + fallback subtensor endpoints**, auto-reconnect on failure.
- **Watchdog / watchtower** auto-restart of every service.
- **Warmup delay after deploy** before sampling (containers report "ready" too
  early; we already hit this — JIT warmup).
- **Single-writer** for the scores/weights tables.
- **Process-local API cache** (TTL ~10min) so dashboard request rate doesn't
  hammer the DB.
- **Deterministic, seeded sampling** keyed by `(window, block, env)` so restarts
  reproduce the same task set.

---

## 10. What this changes in Optima's roadmap

Concretely, to go from "validated harness" to "shippable subnet":

1. **Chain layer** (`optima/chain/`): subtensor connect (+fallback), read
   commitments, sync metagraph, set weights every N blocks. Miner-side commit
   helper. (Use Bittensor's native commit-reveal; drop our custom transport.)
2. **Real DB** behind the `Ledger` interface (Postgres to start): `miners`,
   `scores`, `snapshots`, `system_config`. Single-writer weights.
3. **Service split**: `monitor` (chain→DB), `scheduler` (per-family frontier driver
   over blocks, with champion re-sample on refresh + loser termination),
   `executor` (our `evaluate`, as a pool), `validator` (thin weight-setter),
   `api` (leaderboard).
4. **Functional copy fingerprint** (kernel output hash on canonical inputs) added
   to copy detection.
5. **Isolation**: SSH-control-box / no-secrets-GPU-box / no-egress + per-eval CUDA
   context + watchdog.
6. **Provider abstraction**: rented box now, owned 8×B200 later, one interface.

The selected incentive arithmetic and inactive schema-5 bounty-only state are
implemented and synthetically validated, including D-015 campaign-sized finite
log-relative CROWN debt and bounded reviewed discovery debt. The launch path accepts
one immutable MiniMax-M3 campaign at 100% sizing; historical two-campaign 50/50
cells remain arithmetic research, and no rotation/second-campaign/successor path is
implemented. Target families keep independent clocks without dividing campaign
claim size. The earlier D-012 signer-free
testnet-netuid-307 shadow also
reopened finalized block
7,586,146 (metagraph size 6) and mapped explicitly synthetic states to 850,000 ppm
CROWN, 50,000 ppm discovery, and 100,000 ppm reserve, totaling 1,000,000 ppm
(`submitted=false`; semantic
digest `3dbb3cc27dfd013023c42ba68dd03413d5e5ab1dc8e8626dda3c1a0db18cabaa`, file
SHA-256 `ac695810671cdc6f635a9b30a7fb67f1a885e13bd4fba7e64f2456a08ae88aed`). It
constructed no wallet and supplies no review, settlement, publication, D-015 policy,
or activation authority. The wallet-free atomic one-campaign cutover, schema-6
rollback fence, gapless publication/readback/debit path, and causal audit transport
are implemented but have no live receipt. Production still needs the exact MiniMax-M3
campaign/family/reserve manifests and fresh shadow, independent review/runtime-
invalidity authority, membership-departure history, reliable pending-review-expiry
scheduling, promotion transport/linkage, the production audit GPU canary, and actual
operator activation. The durable bounty lifetime begins at the
retained qualified win, and the pending-review expiry API is landed; the selected
“never both” rule is still policy intent because cross-lane work identity is absent.
A registered-family invalidation API is landed, but it consumes rather than creates
external invalidity authority. Affine remains a working reference for the operational
scaffolding, not evidence that those Optima authorities are complete.

D-015 passed all 14 preregistered screens: 1/10/100-family catalogs had zero
principal dilution; weekly 4.4%/5% normal tapes paid fully with zero expiry or
terminal debt; five-day cadence was marginal and four-day cadence overloaded.
Report semantic digest:
`7975a10b2924330cd527e29b0dfe6f2d9dcb40039f9d8f695b558ec6c6f46590`.

D-014 adds bounded review-delay evidence, not missing infrastructure: its 288-row
cross-architecture replay passed every preregistered 0/1/7-day SLA row with full
discovery payout, zero expiry/unissued debt, no CROWN paid-fraction regression, and
at most 55,555 ppm instantaneous CROWN-capacity dilution; 90/120-day review issued
no stale debt and 30/60/89 days were diagnostic only. It does not supply the external
review service, publication path, activation authority, durable-state completion,
or GPU-performance evidence still called out above.
