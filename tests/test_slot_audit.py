"""In-engine slot audit (optima/audit.py) — unit + dispatcher-wiring tests.

The audit is the fidelity gate that replaced rollout-KL as primary on
launch-nondeterministic arenas (2026-07-07): sampled dispatcher calls re-run the
captured stock baseline on pre-call clones and compare under the slot's verify
tolerances; per-rank rolling receipts feed the eval driver's gate.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from optima import audit, receipts
from optima.dispatch import make_rmsnorm_dispatcher
from optima.registry import Eligibility, KernelImpl, KernelRegistry

SLOT = "norm.rmsnorm"


@pytest.fixture(autouse=True)
def _fresh_audit(monkeypatch):
    monkeypatch.setattr(audit, "_state", {"rate": None, "rng": None})
    monkeypatch.setattr(audit, "_stats", {})
    monkeypatch.delenv("OPTIMA_SLOT_AUDIT", raising=False)
    monkeypatch.delenv("OPTIMA_SLOT_AUDIT_SEED", raising=False)
    monkeypatch.delenv("OPTIMA_SEAM_RECEIPT_DIR", raising=False)


def _arm(monkeypatch, rate="1.0", seed="7"):
    monkeypatch.setenv("OPTIMA_SLOT_AUDIT", rate)
    monkeypatch.setenv("OPTIMA_SLOT_AUDIT_SEED", seed)


# ---- sampling ------------------------------------------------------------------


def test_disabled_without_env():
    assert not audit.enabled()
    assert not audit.sampled()


def test_rate_one_always_samples(monkeypatch):
    _arm(monkeypatch)
    assert audit.enabled()
    assert all(audit.sampled() for _ in range(20))


def test_bad_rate_is_disabled(monkeypatch):
    monkeypatch.setenv("OPTIMA_SLOT_AUDIT", "not-a-number")
    assert not audit.enabled()


def test_seeded_sampling_is_reproducible(monkeypatch):
    # Collective baselines REQUIRE rank-identical decisions: same seed -> same stream.
    _arm(monkeypatch, rate="0.5", seed="123")
    a = [audit.sampled() for _ in range(50)]
    audit._state.update(rate=None, rng=None)  # simulate a second rank, same env
    b = [audit.sampled() for _ in range(50)]
    assert a == b


# ---- record / run --------------------------------------------------------------


def test_record_faithful_no_violation(monkeypatch):
    _arm(monkeypatch)
    x = torch.randn(8, 64)
    audit.record(SLOT, (x,), (x.clone(),))
    s = audit._stats[SLOT]
    assert s["n"] == 1 and s["violations"] == 0 and s["worst_frac"] == 1.0


def test_record_garbage_is_violation(monkeypatch):
    _arm(monkeypatch)
    x = torch.randn(8, 64)
    audit.record(SLOT, (x + 10.0,), (x,))
    s = audit._stats[SLOT]
    assert s["n"] == 1 and s["violations"] == 1 and s["worst_frac"] < 0.5


def test_record_ulp_noise_passes(monkeypatch):
    # A few elements at the tolerance edge must NOT fail an otherwise-faithful kernel
    # (the outlier-channel single-ULP class measured on the v6 stockcheck).
    _arm(monkeypatch)
    x = torch.randn(100, 64)
    y = x.clone()
    y[0, 0] += 100.0  # one wild element out of 6400 -> frac 0.99984 >= 0.995
    audit.record(SLOT, (y,), (x,))
    s = audit._stats[SLOT]
    assert s["violations"] == 0 and s["worst_frac"] < 1.0


def test_record_none_expected_counts_refused(monkeypatch):
    _arm(monkeypatch)
    x = torch.randn(4, 8)
    audit.record(SLOT, (x,), (None,))
    s = audit._stats[SLOT]
    assert s["baseline_refused"] == 1 and s["n"] == 0 and s["violations"] == 0


def test_record_unknown_slot_counts_compare_error(monkeypatch):
    _arm(monkeypatch)
    x = torch.randn(4, 8)
    audit.record("no.such.slot", (x,), (x,))
    assert audit._stats["no.such.slot"]["compare_errors"] == 1


def test_run_baseline_error_is_compare_error_not_crash(monkeypatch):
    _arm(monkeypatch)

    def boom():
        raise RuntimeError("baseline exploded")

    audit.run(SLOT, (torch.randn(4, 8),), boom)
    assert audit._stats[SLOT]["compare_errors"] == 1


def test_run_unwraps_single_tensor_and_tuple(monkeypatch):
    _arm(monkeypatch)
    x = torch.randn(4, 8)
    audit.run(SLOT, (x,), lambda: x.clone())          # bare tensor baseline
    audit.run(SLOT, (x, x), lambda: (x.clone(), x.clone()))  # tuple baseline
    s = audit._stats[SLOT]
    assert s["n"] == 2 and s["violations"] == 0


# ---- receipts ------------------------------------------------------------------


def test_rolling_receipt_overwrites(monkeypatch, tmp_path):
    _arm(monkeypatch)
    monkeypatch.setenv("OPTIMA_SEAM_RECEIPT_DIR", str(tmp_path))
    x = torch.randn(4, 8)
    audit.record(SLOT, (x,), (x.clone(),))
    audit.record(SLOT, (x,), (x.clone(),))
    files = list(tmp_path.glob("audit*.json"))
    assert len(files) == 1  # rolling: same kind+tag+pid file, overwritten
    got = receipts.collect(tmp_path, "audit")
    assert got[0]["n"] == 2 and got[0]["violations"] == 0


# ---- eval-driver gate ----------------------------------------------------------


def test_gate_no_receipts_fails():
    ok, desc = audit.gate([], min_calls=32)
    assert not ok and "no audit receipts" in desc


def test_gate_violations_fail():
    ok, desc = audit.gate([{"n": 100, "violations": 1, "worst_frac": 0.2}], min_calls=32)
    assert not ok and "1 violations" in desc


def test_gate_insufficient_coverage_fails():
    ok, desc = audit.gate([{"n": 5, "violations": 0}], min_calls=32)
    assert not ok and "insufficient coverage" in desc


def test_gate_compare_errors_fail_closed():
    ok, _ = audit.gate([{"n": 100, "violations": 0, "compare_errors": 2}], min_calls=32)
    assert not ok


def test_gate_clean_passes_and_sums_ranks():
    ok, desc = audit.gate(
        [{"n": 20, "violations": 0}, {"n": 20, "violations": 0}], min_calls=32)
    assert ok and "40 audited calls" in desc


def test_gate_requires_exact_slot_by_rank_cartesian_coverage():
    rows = [
        {
            "slot": slot,
            "pid": 100 + rank,
            "rank": rank,
            "world_size": 2,
            "n": 32,
            "violations": 0,
            "compare_errors": 0,
            "worst_frac": 1.0,
        }
        for slot in ("norm.rmsnorm", "activation.silu_and_mul")
        for rank in range(2)
    ]
    slots = ("activation.silu_and_mul", "norm.rmsnorm")
    ok, _ = audit.gate(
        rows,
        min_calls=32,
        expected_slots=slots,
        expected_member_count=2,
    )
    assert ok

    ok, desc = audit.gate(
        rows[:-1],
        min_calls=32,
        expected_slots=slots,
        expected_member_count=2,
    )
    assert not ok and "incomplete" in desc


def test_gate_requires_minimum_calls_on_every_slot_rank_receipt():
    rows = [
        {
            "slot": "norm.rmsnorm",
            "pid": 100 + rank,
            "rank": rank,
            "world_size": 2,
            "n": 31 if rank else 100,
            "violations": 0,
            "compare_errors": 0,
            "worst_frac": 1.0,
        }
        for rank in range(2)
    ]
    ok, desc = audit.gate(
        rows,
        min_calls=32,
        expected_slots=("norm.rmsnorm",),
        expected_member_count=2,
    )
    assert not ok and "per-slot/member coverage" in desc


# ---- dispatcher wiring (rmsnorm: the pure-op case) -------------------------------


def _rmsnorm_ref(x, weight, eps):
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


def _module():
    return SimpleNamespace(variance_epsilon=1e-6,
                           weight=SimpleNamespace(data=torch.ones(64)))


def _baseline_forward(self, x, residual=None, post_residual_addition=None):
    if residual is None:
        return _rmsnorm_ref(x, self.weight.data, self.variance_epsilon)
    added = x + residual
    return _rmsnorm_ref(added, self.weight.data, self.variance_epsilon), added


def _reg(entry):
    reg = KernelRegistry()
    reg.register(KernelImpl(slot=SLOT, bundle_id="t", entry=entry,
                            eligibility=Eligibility(dtypes=frozenset({"float32"}))))
    reg.enable()
    return reg


def test_rmsnorm_dispatcher_faithful_audits_clean(monkeypatch):
    _arm(monkeypatch)

    def entry(x, weight, out, eps):
        out.copy_(_rmsnorm_ref(x, weight, eps))

    d = make_rmsnorm_dispatcher(_baseline_forward, registry=_reg(entry))
    d(_module(), torch.randn(8, 64))
    s = audit._stats[SLOT]
    assert s["n"] == 1 and s["violations"] == 0


def test_rmsnorm_dispatcher_garbage_audited_as_violation(monkeypatch):
    _arm(monkeypatch)

    def entry(x, weight, out, eps):
        out.zero_()  # wrong function

    d = make_rmsnorm_dispatcher(_baseline_forward, registry=_reg(entry))
    d(_module(), torch.randn(8, 64))
    s = audit._stats[SLOT]
    assert s["n"] == 1 and s["violations"] == 1


def test_rmsnorm_dispatcher_fused_path_audits_both_outputs(monkeypatch):
    _arm(monkeypatch)

    def entry(x, weight, out, eps):
        out.copy_(_rmsnorm_ref(x, weight, eps))

    d = make_rmsnorm_dispatcher(_baseline_forward, registry=_reg(entry))
    x, res = torch.randn(8, 64), torch.randn(8, 64)
    out, new_res = d(_module(), x, res)
    s = audit._stats[SLOT]
    assert s["n"] == 1 and s["violations"] == 0
    assert torch.equal(new_res, x + res)


def test_rmsnorm_dispatcher_no_audit_without_env():
    calls = {"n": 0}

    def entry(x, weight, out, eps):
        calls["n"] += 1
        out.copy_(_rmsnorm_ref(x, weight, eps))

    d = make_rmsnorm_dispatcher(_baseline_forward, registry=_reg(entry))
    d(_module(), torch.randn(8, 64))
    assert calls["n"] == 1 and SLOT not in audit._stats


# ---- topk_overlap slots (the msa_prefill selection audit, 2026-07-10) -----------

MSA_SLOT = "attention.msa_prefill_block_score"


def _sel(rows):
    return torch.tensor(rows, dtype=torch.int32).unsqueeze(0)  # (H=1, rows, k)


def test_topk_identical_selection_no_violation(monkeypatch):
    _arm(monkeypatch)
    idx = _sel([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]])
    audit.record(MSA_SLOT, (idx,), (idx.clone(),))
    s = audit._stats[MSA_SLOT]
    assert s["n"] == 1 and s["violations"] == 0 and s["worst_frac"] == 1.0
    assert s["mode"] == "topk_overlap" and s["min_ratio"] == 0.9


def test_topk_disjoint_row_is_violation(monkeypatch):
    # One fully-wrong row of four -> mean overlap 0.75 < the slot's 0.9 floor.
    _arm(monkeypatch)
    base = [[i * 8 + j for j in range(8)] for i in range(4)]
    actual = [row[:] for row in base]
    actual[0] = [100 + j for j in range(8)]
    audit.record(MSA_SLOT, (_sel(actual),), (_sel(base),))
    s = audit._stats[MSA_SLOT]
    assert s["n"] == 1 and s["violations"] == 1
    assert abs(s["worst_frac"] - 0.75) < 1e-6


def test_topk_padding_rows_score_on_valid_entries_only(monkeypatch):
    # -1 fill (short rows) is not a miss: overlap is over the stock row's valid set.
    _arm(monkeypatch)
    expected = _sel([[0, 1, 2, 3, -1, -1, -1, -1]])
    actual = _sel([[0, 1, 2, 3, 9, 10, 11, 12]])
    audit.record(MSA_SLOT, (actual,), (expected,))
    s = audit._stats[MSA_SLOT]
    assert s["n"] == 1 and s["violations"] == 0 and s["worst_frac"] == 1.0


def test_topk_all_invalid_expected_counts_refused(monkeypatch):
    _arm(monkeypatch)
    empty = _sel([[-1] * 8])
    audit.record(MSA_SLOT, (empty.clone(),), (empty,))
    s = audit._stats[MSA_SLOT]
    assert s["n"] == 0 and s["violations"] == 0 and s["baseline_refused"] == 1
