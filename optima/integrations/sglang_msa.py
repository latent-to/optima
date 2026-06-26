"""MSA-indexer seam (attention.msa_block_score) — the finer-than-attention seam for a
SELECTION win (the fp8 MSA indexer on MiniMax-M3).

STATUS: stub. Unlike the five generic seams in ``optima/seams.py``, this chokepoint is
**model-specific** (it only exists when the served model is M3 / an MSA model), so it is NOT
in the generic seam table (the compat canary would false-fail on a non-MSA pin). It belongs
to the M3 *arena*; wire it via ``arenas.py``'s ``seam_adapters`` when that lands, or install
it explicitly when the served model is MSA.

The chokepoint is the MSA backend's decode block-score kernel:
  sglang ``MiniMaxSparseAttnBackend`` -> the per-128-block index score pass (the lightning
  indexer's ``_decode_score_kernel`` in ``.../minimax_sparse_ops/decode/flash_with_topk_idx.py``).

The cheat-resistant split (why the kernel stays upstream of the sampler): the miner kernel
fills the **block scores** only; the validator keeps the **top-k block selection AND the bf16
attend** over the chosen blocks. A wrong score merely mis-selects which blocks are attended —
caught by the ``topk_overlap`` op-gate (the SELECTED set must agree with the bf16 reference)
plus the end-to-end KL. The contract is the ``attention.msa_block_score`` slot
(``optima/slots.py``): ``entry(q, index_k, seq_lens, block_size, out)`` fills
``out:(B, S//block_size)``.

The live install needs a GPU + the M3 sglang backend, so it is built/validated on the pod;
the slot + the ``topk_overlap`` metric (both CPU-tested) are the scorable contract that proves
the subnet can ingest a selection-output win.
"""

from __future__ import annotations

SLOT = "attention.msa_block_score"
CHOKEPOINT = "MiniMaxSparseAttnBackend (decode block-score kernel)"


def install(registry=None) -> None:  # noqa: ARG001 - signature parity with the generic seams
    raise NotImplementedError(
        "MSA-indexer seam is a per-arena (M3) GPU install — wire it via arenas.py seam_adapters "
        "or when the served model is MSA. The scorable contract is the attention.msa_block_score "
        "slot + the topk_overlap metric (CPU-tested in tests/test_msa_block_score.py)."
    )
