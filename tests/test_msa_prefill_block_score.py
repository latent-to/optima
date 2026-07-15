"""CPU tests for the attention.msa_prefill_block_score slot (the prefill-side indexer).

The prefill sibling of attention.msa_block_score: a T-token chunk scores S = prefix+T keys
under the causal rule, emitting a (T, ceil(S/block)) score SHEET gated per row on
topk_overlap. Same selection-not-values philosophy as the decode slot, plus the two failure
classes specific to prefill: a kernel that ignores CAUSALITY (future keys leak into scores)
and a kernel that mis-handles the RAGGED tail block.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from optima.slots import get_slot  # noqa: E402
from optima.registry import eligibility_from_metadata  # noqa: E402
from optima.tensor_spec import validate_output_spec  # noqa: E402
from optima.verify import (  # noqa: E402
    _has_msa_prefill_probe_schema,
    format_verify,
    verify_entry,
)

SLOT = get_slot("attention.msa_prefill_block_score")


def _sheet(q, index_k, prefix_len, scale, block_size, *, causal: bool = True):
    T, D = q.shape
    S = index_k.shape[0]
    s = (q.float() @ index_k.float().t()) * float(scale)
    if causal:
        m = torch.arange(T, device=q.device).view(T, 1)
        n = torch.arange(S, device=q.device).view(1, S)
        s = s.masked_fill(n > int(prefix_len) + m, float("-inf"))
    nblk = (S + block_size - 1) // block_size
    pad = nblk * block_size - S
    if pad:
        s = torch.nn.functional.pad(s, (0, pad), value=float("-inf"))
    return s.view(T, nblk, block_size).amax(-1)


def _faithful(q, index_k, prefix_len, scale, block_size, out):
    out.copy_(_sheet(q, index_k, prefix_len, scale, block_size).to(out.dtype))


def _monotone_perturb(q, index_k, prefix_len, scale, block_size, out):
    # fp8-like: every score moved, selection preserved.
    s = _sheet(q, index_k, prefix_len, scale, block_size)
    out.copy_((s * 1.01 + 0.001).to(out.dtype))


def _wrong_selection(q, index_k, prefix_len, scale, block_size, out):
    s = _sheet(q, index_k, prefix_len, scale, block_size)
    out.copy_((-s).to(out.dtype))


def _acausal(q, index_k, prefix_len, scale, block_size, out):
    # Ignores the causal mask: rows see FUTURE keys. With random data, future blocks
    # outscore visible ones often enough that per-row selections diverge -> must fail.
    out.copy_(_sheet(q, index_k, prefix_len, scale, block_size, causal=False).to(out.dtype))


def _tail_garbage(q, index_k, prefix_len, scale, block_size, out):
    # Correct everywhere except the ragged tail block, which reads past S (modeled as a
    # huge score): the tail block jumps into every row's top-k -> selection disagrees.
    s = _sheet(q, index_k, prefix_len, scale, block_size)
    s[:, -1] = s.max() + 100.0
    out.copy_(s.to(out.dtype))


def _nan_masked_cells(q, index_k, prefix_len, scale, block_size, out):
    s = _sheet(q, index_k, prefix_len, scale, block_size)
    s[s == float("-inf")] = float("nan")
    out.copy_(s.to(out.dtype))


# ---- catalog / contract ------------------------------------------------------

def test_prefill_slot_registered():
    assert SLOT.kind == "block"
    assert SLOT.correctness.mode == "topk_overlap"
    assert SLOT.correctness.top_k == 8
    assert SLOT.kl_threshold == 3e-2


def test_prefill_typed_output_matches_live_score_slab():
    i = SLOT.make_inputs(**SLOT.shapes[0], dtype=torch.bfloat16, device="cpu", seed=0)
    contract = SLOT.output_contract(i)
    assert len(contract.outputs) == 1
    output = contract.outputs[0]
    assert output.shape == SLOT.out_shapes(i)[0]
    assert output.dtype == torch.float32
    assert output.stride_policy == "strided"

    # Model two requests/heads sharing the live [bank,total_q,max_blocks] slab.
    # Each logical output is FP32 and has a row pitch larger than its columns.
    rows, cols = output.shape
    slab = torch.empty((2, rows, cols + 11), dtype=torch.float32)
    for bank in range(2):
        view = slab[bank, :, :cols]
        assert not view.is_contiguous()
        assert view.stride(0) > view.shape[1]
        validate_output_spec(
            contract,
            [view],
            fallback_dtype=torch.bfloat16,
            fallback_device="cpu",
            inputs=(i["q"], i["index_k"]),
        )


def test_out_shape_covers_ragged_tail():
    i = SLOT.make_inputs(**SLOT.shapes[0], dtype=torch.float32, device="cpu", seed=0)
    S = i["index_k"].shape[0]
    assert S % i["block_size"] != 0, "shape must exercise the ragged tail"
    (shape,) = SLOT.out_shapes(i)
    assert shape == (i["q"].shape[0], (S + i["block_size"] - 1) // i["block_size"])


# ---- verify_entry (jittered shapes) ------------------------------------------

def test_prefill_faithful_kernel_verifies():
    res = verify_entry(SLOT, _faithful, dtype=torch.float32, device="cpu", seed=0, jitter_seed=7)
    assert res.passed, res.shape_results
    assert all(r.metric == "overlap" for r in res.shape_results)


def _production_like_eligibility(**overrides):
    capabilities = {
        "dtype": "float32",
        "architecture": "sm103",
        "head_dim": 128,
        "block_size": 128,
        "phase": "prefill",
        "layout": "row_major",
        "graph_mode": "eager",
        "quant": "dense",
    }
    capabilities.update(overrides)
    return eligibility_from_metadata(
        {"graph_safe": False, "capabilities": capabilities}, ("float32",),
        ("sm103",),
    )


def test_prefill_capability_verify_runs_only_in_domain_shapes():
    calls = []

    def counted(*args):
        calls.append((args[0].shape, args[4]))
        _faithful(*args)

    result = verify_entry(
        SLOT,
        counted,
        dtype=torch.float32,
        device="cpu",
        architecture="sm103",
        eligibility=_production_like_eligibility(),
        graph_safe=False,
    )

    assert result.passed, format_verify(result)
    assert result.coverage_required == 1
    assert result.num_applicable == 5
    assert result.num_not_applicable == 6
    assert len(calls) == 5
    assert [r.applicable for r in result.shape_results] == [
        True, True, False, False, True, True, False, False, False, False, True
    ]
    assert all("validator N/A" in r.detail for r in result.shape_results if not r.applicable)
    # The long causality catcher must remain part of the production domain.
    assert result.shape_results[-1].shape["q_len"] == 512
    assert result.shape_results[-1].shape["head_dim"] == 128


def test_prefill_out_of_budget_domain_rejects_without_invocation():
    calls = 0

    def must_not_run(*_args):
        nonlocal calls
        calls += 1
        raise AssertionError("off-domain miner entry was invoked")

    result = verify_entry(
        SLOT,
        must_not_run,
        dtype=torch.float32,
        device="cpu",
        architecture="sm103",
        eligibility=_production_like_eligibility(head_dim=1024),
        graph_safe=False,
    )

    assert not result.passed
    assert calls == 0
    assert result.num_applicable == 0 and not result.coverage_sufficient
    rendered = format_verify(result)
    assert "coverage=0/1" in rendered
    assert rendered.count("\n  N/A shape") == len(SLOT.shapes)


@pytest.mark.parametrize(
    "q_domain, expected_q",
    [
        (256, {256}),
        ({"min": 200, "max": 300}, {200, 250, 300}),
    ],
)
def test_prefill_synthesizes_bounded_probes_inside_new_q_domain(
    q_domain, expected_q
):
    # Synthesis follows the declared score-sheet shape/call schema, not a slot,
    # kernel, or artifact-provider identity.
    renamed_slot = replace(SLOT, name="test.renamed_prefill_score_contract")
    eligibility = _production_like_eligibility(q_len=q_domain)
    assert _has_msa_prefill_probe_schema(
        renamed_slot, eligibility, list(renamed_slot.shapes)
    )
    assert not _has_msa_prefill_probe_schema(
        replace(
            renamed_slot,
            correctness=replace(renamed_slot.correctness, mode="allclose"),
        ),
        eligibility,
        list(renamed_slot.shapes),
    )
    result = verify_entry(
        renamed_slot,
        _faithful,
        dtype=torch.float32,
        device="cpu",
        architecture="sm103",
        eligibility=eligibility,
        graph_safe=False,
        jitter_seed=5,
    )

    assert result.passed, format_verify(result)
    synthesized = [
        r for r in result.shape_results if r.applicable and "prefix_len_override" in r.shape
    ]
    assert synthesized
    assert {r.shape["q_len"] for r in synthesized} == expected_q
    assert any(r.shape.get("causal_probe") is True for r in synthesized)


def test_prefill_acausal_new_exact_q_domain_fails_synthesized_probe():
    result = verify_entry(
        SLOT,
        _acausal,
        dtype=torch.float32,
        device="cpu",
        architecture="sm103",
        eligibility=_production_like_eligibility(q_len=256),
        graph_safe=False,
    )

    assert not result.passed, format_verify(result)
    assert any(
        r.applicable
        and not r.passed
        and r.shape["q_len"] == 256
        and r.shape.get("causal_probe") is True
        for r in result.shape_results
    )


def test_prefill_range_boundary_is_probed_even_when_catalog_intersects():
    def wrong_above_200(q, index_k, prefix_len, scale, block_size, out):
        if q.shape[0] > 200:
            out.zero_()
        else:
            _faithful(q, index_k, prefix_len, scale, block_size, out)

    result = verify_entry(
        SLOT,
        wrong_above_200,
        dtype=torch.float32,
        device="cpu",
        architecture="sm103",
        eligibility=_production_like_eligibility(
            q_len={"min": 16, "max": 300}
        ),
        graph_safe=False,
        jitter_seed=5,
    )

    assert not result.passed, format_verify(result)
    assert any(
        r.applicable and not r.passed and r.shape["q_len"] == 300
        for r in result.shape_results
    )


def test_prefill_finite_one_of_domain_is_not_silently_truncated():
    def wrong_at_d512(q, index_k, prefix_len, scale, block_size, out):
        if q.shape[-1] == 512:
            out.zero_()
        else:
            _faithful(q, index_k, prefix_len, scale, block_size, out)

    result = verify_entry(
        SLOT,
        wrong_at_d512,
        dtype=torch.float32,
        device="cpu",
        architecture="sm103",
        eligibility=_production_like_eligibility(
            q_len=16,
            head_dim=[64, 128, 256, 384, 512],
            block_size=128,
        ),
        graph_safe=False,
    )

    assert not result.passed, format_verify(result)
    assert result.domain_coverage_complete
    assert {r.shape["head_dim"] for r in result.shape_results if r.applicable} >= {
        64, 128, 256, 384, 512
    }
    assert any(
        r.applicable and not r.passed and r.shape["head_dim"] == 512
        for r in result.shape_results
    )


def test_prefill_oversized_finite_cross_product_fails_instead_of_truncating():
    result = verify_entry(
        SLOT,
        _faithful,
        dtype=torch.float32,
        device="cpu",
        architecture="sm103",
        eligibility=_production_like_eligibility(
            head_dim=[32, 64, 96, 128, 160, 192, 224, 256, 288],
            block_size=[64, 128],
            q_len=[16, 32, 48, 64],
        ),
        graph_safe=False,
    )

    assert not result.passed
    assert not result.domain_coverage_complete
    assert "cross-product" in result.domain_coverage_detail


def test_prefill_each_finite_block_size_gets_its_own_nonvacuous_prefix():
    calls = []

    def wrong_at_b4096(q, index_k, prefix_len, scale, block_size, out):
        calls.append((int(block_size), int(prefix_len)))
        if block_size == 4096:
            out.zero_()
        else:
            _faithful(q, index_k, prefix_len, scale, block_size, out)

    result = verify_entry(
        SLOT,
        wrong_at_b4096,
        dtype=torch.float32,
        device="cpu",
        architecture="sm103",
        eligibility=_production_like_eligibility(
            q_len=16,
            head_dim=128,
            block_size=[64, 4096],
        ),
        graph_safe=False,
    )

    assert not result.passed, format_verify(result)
    assert result.domain_coverage_complete
    assert {block_size for block_size, _prefix in calls} == {64, 4096}
    assert all(
        (prefix_len + block_size) // block_size > SLOT.correctness.top_k
        for block_size, prefix_len in calls
    )
    assert any(
        r.applicable and not r.passed and r.shape["block_size"] == 4096
        for r in result.shape_results
    )


def test_prefill_unaffordable_finite_block_combination_fails_closed():
    result = verify_entry(
        SLOT,
        _faithful,
        dtype=torch.float32,
        device="cpu",
        architecture="sm103",
        eligibility=_production_like_eligibility(
            q_len=64,
            head_dim=128,
            block_size=[64, 4096],
        ),
        graph_safe=False,
    )

    assert not result.passed
    assert not result.domain_coverage_complete
    assert "per-shape work limit" in result.domain_coverage_detail


@pytest.mark.parametrize(
    "legacy_metadata",
    [
        {"max_last_dim": 1024},
        {"max_num_tokens": 2000},
        {"min_num_tokens": 16},
    ],
)
def test_prefill_legacy_domain_outside_probe_budget_fails_closed(
    legacy_metadata,
):
    eligibility = eligibility_from_metadata(
        {"graph_safe": False, **legacy_metadata},
        ("float32",),
        ("sm103",),
    )
    result = verify_entry(
        SLOT,
        _faithful,
        dtype=torch.float32,
        device="cpu",
        architecture="sm103",
        eligibility=eligibility,
        graph_safe=False,
    )

    assert not result.passed
    assert not result.domain_coverage_complete
    assert "bounded probe limits" in result.domain_coverage_detail


def test_prefill_narrow_domain_reports_coverage_without_pooling_variants():
    calls = []

    def counted(*args):
        calls.append(args[0].shape[0])
        _faithful(*args)

    result = verify_entry(
        SLOT,
        counted,
        dtype=torch.float32,
        device="cpu",
        architecture="sm103",
        eligibility=_production_like_eligibility(q_len=16),
        graph_safe=False,
        jitter_seed=7,
    )

    assert result.passed
    assert result.num_applicable == 2
    assert result.num_failed == 0
    assert result.coverage_sufficient
    assert calls == [16, 16]


def test_prefill_causal_probe_survives_cli_jitter_path():
    result = verify_entry(
        SLOT,
        _faithful,
        dtype=torch.float32,
        device="cpu",
        architecture="sm103",
        eligibility=_production_like_eligibility(q_len=128),
        graph_safe=False,
        jitter_seed=0,
    )

    assert result.passed, format_verify(result)
    assert result.num_applicable == 2


def test_prefill_causal_probe_varies_declared_graph_inputs_across_seeds():
    shape = next(shape for shape in SLOT.shapes if shape.get("causal_probe") is True)
    first = SLOT.make_inputs(dtype=torch.bfloat16, device="cpu", seed=1, **shape)
    second = SLOT.make_inputs(dtype=torch.bfloat16, device="cpu", seed=2, **shape)

    assert SLOT.graph_dynamic_inputs == ("q", "index_k")
    assert all(
        not torch.equal(first[name], second[name])
        for name in SLOT.graph_dynamic_inputs
    )


def test_prefill_verify_exercises_fp32_padded_output():
    observed = []

    def stride_aware(q, index_k, prefix_len, scale, block_size, out):
        observed.append((out.dtype, out.is_contiguous(), out.shape, out.stride()))
        _faithful(q, index_k, prefix_len, scale, block_size, out)

    res = verify_entry(
        SLOT,
        stride_aware,
        dtype=torch.bfloat16,
        device="cpu",
        seed=0,
        shapes=[SLOT.shapes[0]],
        graph_safe=False,
    )
    assert res.passed, res.shape_results
    assert observed
    dtype, contiguous, shape, stride = observed[0]
    assert dtype == torch.float32
    assert not contiguous
    assert stride[-1] == 1
    assert stride[-2] > shape[-1]


def test_prefill_verify_rejects_contiguous_bf16_output_assumption():
    def contiguous_bf16_only(q, index_k, prefix_len, scale, block_size, out):
        if out.dtype != torch.bfloat16 or not out.is_contiguous():
            raise RuntimeError("kernel assumed a contiguous BF16 score sheet")
        _faithful(q, index_k, prefix_len, scale, block_size, out)

    res = verify_entry(
        SLOT,
        contiguous_bf16_only,
        dtype=torch.bfloat16,
        device="cpu",
        seed=0,
        shapes=[SLOT.shapes[0]],
        graph_safe=False,
    )
    assert not res.passed
    assert "contiguous BF16" in res.shape_results[0].detail


def test_prefill_monotone_perturbation_verifies():
    res = verify_entry(SLOT, _monotone_perturb, dtype=torch.float32, device="cpu", seed=0)
    assert res.passed, res.shape_results


def test_prefill_wrong_selection_fails():
    res = verify_entry(SLOT, _wrong_selection, dtype=torch.float32, device="cpu", seed=0)
    assert not res.passed


def test_prefill_acausal_kernel_fails():
    res = verify_entry(SLOT, _acausal, dtype=torch.float32, device="cpu", seed=0)
    assert not res.passed


@pytest.mark.parametrize(
    ("q_len", "head_dim", "block_size"),
    [(16, 128, 128), (128, 128, 128), (33, 64, 128), (64, 128, 64)],
)
def test_prefill_acausal_exact_short_shape_specialization_fails(
    q_len, head_dim, block_size
):
    res = verify_entry(
        SLOT,
        _acausal,
        dtype=torch.float32,
        device="cpu",
        seed=0,
        architecture="sm103",
        eligibility=_production_like_eligibility(
            q_len=q_len, head_dim=head_dim, block_size=block_size
        ),
        graph_safe=False,
    )
    assert not res.passed, format_verify(res)
    assert any(
        r.applicable
        and not r.passed
        and r.shape["q_len"] == q_len
        and r.shape.get("causal_probe") is True
        for r in res.shape_results
    )


@pytest.mark.parametrize(
    ("head_dim", "block_size"),
    [(64, 128), (128, 64), (128, 128)],
)
def test_prefill_acausal_unbounded_q_shape_specialization_fails(
    head_dim, block_size
):
    res = verify_entry(
        SLOT,
        _acausal,
        dtype=torch.float32,
        device="cpu",
        seed=0,
        architecture="sm103",
        eligibility=_production_like_eligibility(
            head_dim=head_dim, block_size=block_size
        ),
        graph_safe=False,
    )
    assert not res.passed, format_verify(res)
    assert any(
        r.applicable and not r.passed and r.shape["q_len"] == 512
        for r in res.shape_results
    )


def test_prefill_tail_block_garbage_fails():
    res = verify_entry(SLOT, _tail_garbage, dtype=torch.float32, device="cpu", seed=0)
    assert not res.passed


def test_prefill_nan_in_masked_cells_fails_instead_of_ranking_as_minus_inf():
    res = verify_entry(
        SLOT,
        _nan_masked_cells,
        dtype=torch.float32,
        device="cpu",
        seed=0,
        shapes=[SLOT.shapes[-1]],
        graph_safe=False,
    )
    assert not res.passed
    assert "NaN or +inf" in res.shape_results[0].detail


# ---- the gate is never vacuous, per ROW --------------------------------------

def test_prefill_gate_is_never_vacuous():
    # Row 0 is the worst case (it sees only prefix_len+1 keys): even under count-dim
    # jitter driving prefix_blocks down, every row's VISIBLE block count must exceed
    # top_k, else top-k-of-k makes that row's overlap 1.0 for any output.
    for sh in list(SLOT.shapes) + [dict(SLOT.shapes[0], prefix_blocks=1),
                                   dict(SLOT.shapes[0], q_len=1)]:
        i = SLOT.make_inputs(**sh, dtype=torch.float32, device="cpu", seed=0)
        visible_row0 = (int(i["prefix_len"]) + 1 + i["block_size"] - 1) // i["block_size"]
        assert visible_row0 > SLOT.correctness.top_k, f"vacuous row-0: {sh} -> {visible_row0}"
