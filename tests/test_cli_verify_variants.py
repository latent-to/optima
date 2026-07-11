"""CLI qualification must preserve per-variant metadata and graph policy."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from optima import cli
from optima.verify import VerifyResult


def _write_variant_bundle(tmp_path):
    (tmp_path / "kernels").mkdir()
    (tmp_path / "metadata").mkdir()
    (tmp_path / "kernels" / "small.py").write_text("def run(*args):\n    return None\n")
    (tmp_path / "kernels" / "large.py").write_text("def run(*args):\n    return None\n")

    small = {
        "graph_safe": False,
        "capabilities": {"q_len": {"max": 1}, "phase": ["prefill"]},
    }
    large = {
        "graph_safe": True,
        "capabilities": {"q_len": {"min": 2}, "phase": ["prefill"]},
    }
    (tmp_path / "metadata" / "small.json").write_text(json.dumps(small))
    (tmp_path / "metadata" / "large.json").write_text(json.dumps(large))
    (tmp_path / "manifest.toml").write_text(
        'bundle_id = "cli-variant-qualification"\n'
        'abi_version = "optima-op-abi-v0"\n\n'
        '[[ops]]\n'
        'slot = "attention.msa_prefill_block_score"\n'
        'variant = "small"\n'
        'source = "kernels/small.py"\n'
        'entry = "run"\n'
        'dtypes = ["float16"]\n'
        'architectures = ["sm_120"]\n'
        'metadata = "metadata/small.json"\n\n'
        '[[ops]]\n'
        'slot = "attention.msa_prefill_block_score"\n'
        'variant = "large"\n'
        'source = "kernels/large.py"\n'
        'entry = "run"\n'
        'dtypes = ["bfloat16"]\n'
        'architectures = ["sm_100"]\n'
        'metadata = "metadata/large.json"\n'
    )
    return small, large


def test_cmd_verify_forwards_each_variant_with_its_declared_domain(
    tmp_path, monkeypatch
):
    small, large = _write_variant_bundle(tmp_path)
    calls = []

    monkeypatch.setattr(cli, "_recursive_scan_ok", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        cli, "scan_path", lambda path: SimpleNamespace(ok=True, violations=[])
    )

    def fake_call_in_subprocess(target, *args, **kwargs):
        calls.append((target, args, kwargs))
        return VerifyResult(
            slot=args[0], dtype=kwargs["dtype_name"], passed=True, shape_results=[]
        )

    import optima.eval._launch as launch

    monkeypatch.setattr(launch, "call_in_subprocess", fake_call_in_subprocess)
    args = argparse.Namespace(
        bundle=str(tmp_path),
        dtype="float16",
        device="cpu",
        seed=7,
        model=None,
        world_size=None,
        tp_size=None,
    )

    assert cli.cmd_verify(args) == 0
    assert len(calls) == 2

    _, small_args, small_kwargs = calls[0]
    _, large_args, large_kwargs = calls[1]
    assert small_args[1].endswith("kernels/small.py")
    assert large_args[1].endswith("kernels/large.py")
    assert small_kwargs["eligibility_metadata"] == small
    assert large_kwargs["eligibility_metadata"] == large
    assert small_kwargs["manifest_dtypes"] == ("float16",)
    assert large_kwargs["manifest_dtypes"] == ("bfloat16",)
    assert small_kwargs["manifest_architectures"] == ("sm_120",)
    assert large_kwargs["manifest_architectures"] == ("sm_100",)
    assert small_kwargs["graph_safe"] is False
    assert large_kwargs["graph_safe"] is True
    assert small_kwargs["tp_size"] is None
    assert large_kwargs["world_size"] is None


def test_cmd_verify_runs_two_shape_variants_through_real_verifier(
    tmp_path, monkeypatch
):
    (tmp_path / "kernels").mkdir()
    (tmp_path / "metadata").mkdir()
    source = (
        "import torch\n\n"
        "def run(q, index_k, prefix_len, scale, block_size, out):\n"
        "    scores = (q.float() @ index_k.float().t()) * float(scale)\n"
        "    rows = torch.arange(q.shape[0], device=q.device).view(-1, 1)\n"
        "    keys = torch.arange(index_k.shape[0], device=q.device).view(1, -1)\n"
        "    scores = scores.masked_fill(keys > int(prefix_len) + rows, float('-inf'))\n"
        "    blocks = (index_k.shape[0] + block_size - 1) // block_size\n"
        "    pad = blocks * block_size - index_k.shape[0]\n"
        "    if pad:\n"
        "        scores = torch.nn.functional.pad(scores, (0, pad), value=float('-inf'))\n"
        "    out.copy_(scores.view(q.shape[0], blocks, block_size).amax(-1))\n"
    )
    (tmp_path / "kernels" / "score.py").write_text(source)
    rows = []
    variants = (
        ("q16", 16, "float32"),
        ("q128", 128, "float32"),
        ("q16-fp16", 16, "float16"),
    )
    for variant, q_len, dtype in variants:
        metadata = {
            "graph_safe": False,
            "capabilities": {
                "dtype": dtype,
                "head_dim": 128,
                "block_size": 128,
                "q_len": q_len,
                "phase": "prefill",
                "layout": "row_major",
                "graph_mode": "eager",
                "quant": "dense",
                "top_k": 8,
                "tp_size": 4,
                "world_size": 8,
            },
        }
        (tmp_path / "metadata" / f"{variant}.json").write_text(
            json.dumps(metadata)
        )
        rows.append(
            "[[ops]]\n"
            'slot = "attention.msa_prefill_block_score"\n'
            f'variant = "{variant}"\n'
            'source = "kernels/score.py"\n'
            'entry = "run"\n'
            f'dtypes = ["{dtype}"]\n'
            f'metadata = "metadata/{variant}.json"\n'
        )
    (tmp_path / "manifest.toml").write_text(
        'bundle_id = "cli-real-variants"\n'
        'abi_version = "optima-op-abi-v0"\n\n'
        + "\n".join(rows)
    )

    monkeypatch.setattr(cli, "_recursive_scan_ok", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        cli, "scan_path", lambda path: SimpleNamespace(ok=True, violations=[])
    )
    results = []

    import optima.eval._launch as launch

    real_call_in_subprocess = launch.call_in_subprocess

    def run_observed_subprocess(target, *args, **kwargs):
        result = real_call_in_subprocess(target, *args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(launch, "call_in_subprocess", run_observed_subprocess)
    args = argparse.Namespace(
        bundle=str(tmp_path),
        dtype="float32",
        device="cpu",
        seed=7,
        model=None,
        world_size=8,
        tp_size=4,
    )

    assert cli.cmd_verify(args) == 0
    assert len(results) == 3
    assert all(result.passed for result in results[:2])
    assert all(result.num_applicable >= 2 for result in results[:2])
    assert results[2].context_inapplicable
    assert results[2].num_applicable == 0

    # A TP/world-constrained bundle with no matching arena context is not a
    # successful verification merely because every row was reported N/A.
    results.clear()
    args.tp_size = None
    args.world_size = None

    def run_inline(target, *call_args, **kwargs):
        result = target(*call_args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(launch, "call_in_subprocess", run_inline)
    assert cli.cmd_verify(args) == 2
    assert len(results) == 3
    assert all(result.context_inapplicable for result in results)


def test_cmd_verify_rejects_malformed_normative_metadata(tmp_path, monkeypatch):
    _write_variant_bundle(tmp_path)
    for name in ("small", "large"):
        (tmp_path / "metadata" / f"{name}.json").write_text(
            json.dumps({"graph_safe": "false"})
        )
    monkeypatch.setattr(cli, "_recursive_scan_ok", lambda *args, **kwargs: True)
    args = argparse.Namespace(
        bundle=str(tmp_path),
        dtype="float16",
        device="cpu",
        seed=7,
        model=None,
        world_size=None,
        tp_size=None,
    )

    assert cli.cmd_verify(args) == 2


def test_cmd_verify_rejects_overlapping_variants_before_candidate_invocation(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "kernels").mkdir()
    (tmp_path / "metadata").mkdir()
    (tmp_path / "kernels" / "score.py").write_text(
        "import torch\n\n"
        "def run(q, index_k, prefix_len, scale, block_size, out):\n"
        "    scores = (q.float() @ index_k.float().t()) * float(scale)\n"
        "    rows = torch.arange(q.shape[0], device=q.device).view(-1, 1)\n"
        "    keys = torch.arange(index_k.shape[0], device=q.device).view(1, -1)\n"
        "    scores = scores.masked_fill(keys > int(prefix_len) + rows, float('-inf'))\n"
        "    blocks = (index_k.shape[0] + block_size - 1) // block_size\n"
        "    pad = blocks * block_size - index_k.shape[0]\n"
        "    if pad:\n"
        "        scores = torch.nn.functional.pad(scores, (0, pad), value=float('-inf'))\n"
        "    out.copy_(scores.view(q.shape[0], blocks, block_size).amax(-1))\n"
    )
    for variant, q_domain in (
        ("left", {"min": 1, "max": 128}),
        ("right", {"min": 128, "max": 256}),
    ):
        (tmp_path / "metadata" / f"{variant}.json").write_text(
            json.dumps(
                {
                    "graph_safe": False,
                    "capabilities": {
                        "dtype": "float32",
                        "q_len": q_domain,
                        "phase": "prefill",
                    },
                }
            )
        )
    (tmp_path / "manifest.toml").write_text(
        'bundle_id = "cli-overlap-preflight"\n'
        'abi_version = "optima-op-abi-v0"\n\n'
        '[[ops]]\n'
        'slot = "attention.msa_prefill_block_score"\n'
        'variant = "left"\n'
        'source = "kernels/score.py"\n'
        'entry = "run"\n'
        'dtypes = ["float32"]\n'
        'metadata = "metadata/left.json"\n\n'
        '[[ops]]\n'
        'slot = "attention.msa_prefill_block_score"\n'
        'variant = "right"\n'
        'source = "kernels/score.py"\n'
        'entry = "run"\n'
        'dtypes = ["float32"]\n'
        'metadata = "metadata/right.json"\n'
    )

    monkeypatch.setattr(cli, "_recursive_scan_ok", lambda *args, **kwargs: True)
    candidate_calls = 0

    import optima.eval._launch as launch

    def must_not_invoke_candidate(*_args, **_kwargs):
        nonlocal candidate_calls
        candidate_calls += 1
        raise AssertionError("overlapping variants reached candidate verification")

    monkeypatch.setattr(launch, "call_in_subprocess", must_not_invoke_candidate)
    args = argparse.Namespace(
        bundle=str(tmp_path),
        dtype="float32",
        device="cpu",
        seed=7,
        model=None,
        world_size=None,
        tp_size=None,
    )

    assert cli.cmd_verify(args) == 2
    assert candidate_calls == 0
    output = capsys.readouterr().out
    assert "variant='right'" in output
    assert "overlapping capability domains" in output
    assert "'left' and 'right'" in output
