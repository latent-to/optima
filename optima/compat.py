"""sglang compatibility canary — assert our integration points survive an upgrade.

Our harness patches sglang internals (the `SiluAndMul` / `RMSNorm` seams, the
`MultiPlatformOp` base, the Engine logprob API, specific `ServerArgs` kwargs). Any
sglang upgrade can move those. This canary introspects the INSTALLED sglang —
imports + signatures only, **no GPU, no model** — and checks every seam and API we
depend on still exists.

Run `optima compat` after bumping sglang. If it goes red, the seams need an
adapter before that version can be used for scoring. (A green canary is necessary
but not sufficient — the runtime smoke test, "broken kernel still FAILs the gate,"
is the behavioral confirmation on the pod.)
"""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass

# The sglang version scored against. Bump DELIBERATELY and in a coordinated way —
# see docs/SGLANG_TRACKING.md. All validators must run the same version (consensus).
#
# 0.5.13.post1 (CUDA 13). VALIDATION STATE — read before treating as authoritative:
#   * static seam canary: GREEN — all chokepoints intact on 0.5.13.post1 (sglang-canary CI,
#     2026-06-22: FusedMoE/RadixAttention/SiluAndMul/RMSNorm/all_reduce + logprob API + ServerArgs).
#   * GPU end-to-end re-validation (broken-bundle gate FAILs, faithful kernels PASS) + champion
#     RE-BASELINE: **PENDING** — run on a pod before this pin is authoritative for scoring.
#     0.5.12.post1 was the last fully H100-validated pin.
PINNED_SGLANG = "0.5.13.post1"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def run_checks() -> list[Check]:
    checks: list[Check] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append(Check(name, bool(ok), str(detail)))

    try:
        import sglang
    except Exception as exc:  # noqa: BLE001
        add("import sglang", False, repr(exc))
        return checks

    ver = getattr(sglang, "__version__", "?")
    add(
        f"sglang installed (pinned {PINNED_SGLANG})",
        True,
        f"found {ver}" + ("" if ver == PINNED_SGLANG else "  <-- DIFFERS from pin"),
    )

    # Table-driven baseline: every adapter in the single seam table (optima/seams.py)
    # must have its target module import and its Class.method chokepoint present. Adding
    # a seam to that table auto-adds this canary (no separate edit here). The bespoke
    # signature checks below enrich these for the known seams.
    import importlib

    from optima.seams import SEAM_ADAPTERS

    for adapter in SEAM_ADAPTERS:
        cls_name, _, meth = adapter.chokepoint.partition(".")
        try:
            mod = importlib.import_module(adapter.target_module)
            cls = getattr(mod, cls_name, None)
            ok = cls is not None and hasattr(cls, meth)
            add(f"seam table: {adapter.name} ({adapter.chokepoint})", ok,
                "" if ok else f"missing {adapter.chokepoint} in {adapter.target_module}")
        except Exception as exc:  # noqa: BLE001
            add(f"seam table: {adapter.name} ({adapter.chokepoint})", False, repr(exc))

    try:
        from sglang.srt.layers.utils.multi_platform import MultiPlatformOp
        mpo = MultiPlatformOp
        add("MultiPlatformOp base present", True)
    except Exception as exc:  # noqa: BLE001
        mpo = None
        add("MultiPlatformOp base present", False, repr(exc))

    # activation seam (SiluAndMul slot)
    try:
        from sglang.srt.layers.activation import SiluAndMul

        ok = hasattr(SiluAndMul, "forward_cuda") and hasattr(SiluAndMul, "forward_native")
        if mpo is not None:
            ok = ok and issubclass(SiluAndMul, mpo)
        add("seam: SiluAndMul (activation)", ok, "needs forward_cuda/native on a MultiPlatformOp")
    except Exception as exc:  # noqa: BLE001
        add("seam: SiluAndMul (activation)", False, repr(exc))

    # norm seam (RMSNorm slot)
    try:
        from sglang.srt.layers.layernorm import RMSNorm

        params = list(inspect.signature(RMSNorm.forward_cuda).parameters)
        ok = hasattr(RMSNorm, "forward_cuda") and "residual" in params
        if mpo is not None:
            ok = ok and issubclass(RMSNorm, mpo)
        add("seam: RMSNorm (layernorm)", ok, f"forward_cuda params={tuple(params)}")
    except Exception as exc:  # noqa: BLE001
        add("seam: RMSNorm (layernorm)", False, repr(exc))

    # attention seam (the attention BLOCK slot chokepoint: RadixAttention.forward)
    try:
        from sglang.srt.layers.radix_attention import RadixAttention

        params = set(inspect.signature(RadixAttention.forward).parameters)
        ok = hasattr(RadixAttention, "forward") and {"q", "k", "v", "forward_batch"} <= params
        add("seam: RadixAttention (attention)", ok, f"forward params={tuple(sorted(params))}")
    except Exception as exc:  # noqa: BLE001
        add("seam: RadixAttention (attention)", False, repr(exc))

    # MoE seam (the MoE BLOCK slot chokepoint: FusedMoE.forward(hidden_states, topk_output))
    try:
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        params = set(inspect.signature(FusedMoE.forward).parameters)
        ok = hasattr(FusedMoE, "forward") and {"hidden_states", "topk_output"} <= params
        add("seam: FusedMoE (moe.fused_experts)", ok, f"forward params={tuple(sorted(params))}")
    except Exception as exc:  # noqa: BLE001
        add("seam: FusedMoE (moe.fused_experts)", False, repr(exc))

    # collective seam (the TP-comms chokepoint: GroupCoordinator.all_reduce)
    try:
        from sglang.srt.distributed.parallel_state import GroupCoordinator

        params = set(inspect.signature(GroupCoordinator.all_reduce).parameters)
        ok = hasattr(GroupCoordinator, "all_reduce") and "input_" in params
        add("seam: GroupCoordinator.all_reduce (collective)", ok, f"all_reduce params={tuple(sorted(params))}")
    except Exception as exc:  # noqa: BLE001
        add("seam: GroupCoordinator.all_reduce (collective)", False, repr(exc))

    # Engine logprob API (we read top-k logprobs for KL)
    try:
        gp = set(inspect.signature(sglang.Engine.generate).parameters)
        need = {"prompt", "sampling_params", "return_logprob", "logprob_start_len", "top_logprobs_num"}
        add("Engine.generate logprob API", need <= gp, f"missing: {sorted(need - gp) or 'none'}")
    except Exception as exc:  # noqa: BLE001
        add("Engine.generate logprob API", False, repr(exc))

    # ServerArgs kwargs we pass to Engine(...)
    try:
        from sglang.srt.server_args import ServerArgs

        fields = {f.name for f in dataclasses.fields(ServerArgs)}
        need = {
            "model_path", "dtype", "attention_backend", "disable_cuda_graph",
            "mem_fraction_static", "enable_deterministic_inference", "random_seed", "log_level",
        }
        add("ServerArgs accepts our kwargs", need <= fields, f"missing: {sorted(need - fields) or 'none'}")
    except Exception as exc:  # noqa: BLE001
        add("ServerArgs accepts our kwargs", False, repr(exc))

    return checks


def format_checks(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        mark = "ok  " if c.ok else "FAIL"
        lines.append(f"  [{mark}] {c.name}" + (f"  — {c.detail}" if c.detail else ""))
    n_fail = sum(1 for c in checks if not c.ok)
    lines.append("")
    lines.append(
        "ALL SEAMS INTACT" if n_fail == 0
        else f"{n_fail} CHECK(S) FAILED — seams need an adapter before scoring on this sglang"
    )
    return "\n".join(lines)
