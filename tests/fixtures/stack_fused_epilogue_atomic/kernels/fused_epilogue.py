try:
    import fused_epilogue_sm103 as _ext
except ImportError:
    _ext = None


def ar_residual_rmsnorm(x, residual, weight, eps, out_norm, out_residual, group):
    if _ext is None:
        out_residual.copy_(x + residual)
        out_norm.copy_(out_residual)
        return
    _ext.ar_residual_rmsnorm(x, residual, weight, eps, out_norm, out_residual, group)


def moe_finalize_ar_rmsnorm(
    gemm2_out,
    residual,
    weight,
    eps,
    out_norm,
    out_residual,
    group,
):
    if _ext is None:
        out_residual.copy_(gemm2_out + residual)
        out_norm.copy_(out_residual)
        return
    _ext.moe_finalize_ar_rmsnorm(
        gemm2_out, residual, weight, eps, out_norm, out_residual, group
    )
