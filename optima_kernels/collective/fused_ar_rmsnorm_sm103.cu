// fused_epilogue_sm103.cu — MiniMax-M3 decode epilogue fusion, TP2/TP4/TP8, Blackwell SM103.
//
// TWO ENTRY POINTS, ONE CORE (measured surface: 20% of decode GPU, all launch-latency-bound):
//   1. ar_add_rmsnorm            : one-shot AR + residual-add + RMSNorm            (attn-side, 8.4%)
//   2. moe_finalize_ar_add_rmsnorm: MoE-finalize prologue (+shared-expert add) + the same core
//                                                                                  (MoE-side, 11.6%)
// Replaces per chain: [finalizeMoeRouting] -> [all_reduce_one_shot] -> [fused_add_rmsnorm]
// (3 launches + 2 full hidden-states HBM round-trips) with ONE kernel.
//
// PROVENANCE / DONORS (architecture adapted, implementation ours):
//   - flashinfer include/flashinfer/comm/trtllm_moe_allreduce_fusion.cuh (finalize-as-AR-prologue,
//     one-shot Lamport). v1 used a sequence-flag protocol instead of Lamport -0.0 sentinels: no
//     payload-value restrictions, no triple-buffer clear pass; costs one extra flag store/spin.
//   - UPDATE 2026-07-02: BOTH exchange protocols are now compiled in, selected by the host-side
//     `lamport` flag (A/B-able at launch). The LAMPORT variant makes the DATA its own arrival
//     signal: bf16 -0.0 (0x8000) = not-arrived sentinel; real -0.0 payloads are flushed to +0.0
//     before hitting the wire; comm slots are TRIPLE-BUFFERED, rotated buf = counter % 3 by the
//     same device-side per-slot counter, and the PREVIOUS buffer is re-cleared to sentinel by the
//     SAME kernel after its reduce. This kills the per-thread __threadfence_system + flag
//     store/spin of the seq path entirely (no fences, no flags — arrival == payload visible).
//     Architecture is the TRT-LLM/flashinfer production pattern; implementation is ours.
//   - vLLM/sglang custom_all_reduce (push one-shot, per-block device-side flag progression).
//   - TRT-LLM moe_kernels.cu finalizeMoeRoutingKernel (token-major gather + weighted-sum semantics,
//     via sglang jit moe_finalize_fuse_shared's index convention: idx[t*K+k] -> permuted row, -1 = skip).
//
// FIXES vs the earlier TP2 prototype (kernels/fused_ar_rmsnorm_sm103.cu) — both were replay-killers:
//   a) NON-ROTATING FLAGS: READY/IDLE toggling lets a fast rank at replay i+1 consume the peer's
//      still-READY flag from replay i (stale partials). Here every invocation uses a fresh MONOTONIC
//      SEQUENCE value (per token-slot device counter, advanced inside the kernel -> graph-replay-safe,
//      wrap-safe via exact == compare; all ranks replay the same graphs so sequences stay in lockstep).
//   b) .gpu-SCOPE FENCES: ld.acquire.gpu / st.release.gpu only order within ONE GPU. Peer-visible
//      data/flag ordering over NVLink needs .sys scope. All protocol ops here are .sys.
//
// PROTOCOL (push one-shot, per token-slot t, my rank = m, world = R):
//   1. compute my partial row p_m[t]  (attn entry: given; moe entry: finalize prologue)
//   2. store p_m[t] into slot (t, m) of EVERY rank's data buffer (R-1 remote NVLink stores + 1 local)
//   3. st.release.sys seq  into every rank's flag[t][m]           (seq = my counter[t] + 1)
//   4. spin ld.acquire.sys on MY LOCAL flag[t][r] == seq for all r (local polling only)
//   5. full[t] = sum_r p_r[t]  (fp32 accum, read from MY LOCAL data slots)
//   6. residual_out = full + residual;  norm_out = rmsnorm(residual_out) * (weight_bias + w)
//   7. counter[t] = seq  (thread 0; next invocation uses seq+1)
//   NOTE step 5 reads local slots that peers pushed -> no remote reads anywhere; spins are local.
//
// CONSTRAINTS (documented, enforced):
//   - one-shot only: comm bytes = N * H * 2 per rank per step; decode-sized (N <= MAX_SLOTS=1024).
//     Prefill / large N must take the stock path (dispatch handled by the Python wrapper).
//   - cross-rank sum accumulates in fp32 (better than the stock bf16 one-shot push AR), but
//     reduction ORDER differs from NCCL -> validate vs the measured stock-vs-stock KL noise floor,
//     never claim bit-exactness.
//   - Norm supports weight_bias (0.0 = plain gamma; 1.0 = Gemma-style (1+gamma)). M3's variant is
//     RECONCILED AT INTEGRATION (v0 check) — recon found conflicting claims; both are one flag here.
//   - buffers MUST come from a non-VMM allocator (cudaMalloc / default torch caching allocator).
//     PYTORCH_CUDA_ALLOC_CONF=expandable_segments breaks cudaIpcGetMemHandle (learned the hard way).
//
// v7 (2026-07-02, rung-3 S2): COLUMN-WINDOW consumers for the fe_chunk'd gemm2 producer —
//   moe_epilogue_ptrs_win consumes ONE packed D plane per call (col_off/col_w window),
//   shared Lamport slot with advance-flag (last chunk only), partial-ssq slots
//   [MAX_SLOTS, MAX_NC] + last-consume rinv + full-row norm, SPLIT prev-buffer clear.
//   Spec: experiments/minimax_m3/frontier_2026-07-02/07_GEMM2_AR_OVERLAP_PLAN.md §5 R1/§7 S2.
//
// BUILD: build.sh (nvcc -arch=sm_103a for B300; add sm_100a for B200), torch-extension .so.
// TEST : bench_tp4.py (torchrun, fp64 oracle, --graphs capture/replay) + replay stress (500+ replays);
//        v7: bench_tp4_chunk.py (TP4 chunk correctness/stress) + bench_hermetic --chain deepchunk.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace fused_epilogue {

constexpr int H = 6144;           // M3 hidden (bf16) — templated tile math assumes this
// v3 (2026-07-02): THREADS 256 -> 768 = H/VEC, i.e. ONE 16B chunk per thread. Measured v2
// (256 thr x 3 chunks) at 23.5-24.6 us graphed T=64 vs stock 19.9 / flashinfer-fused 17.8:
// the per-thread latency chain (push 3x3 chunks, spin 9 chunks SEQUENTIALLY, 24-elem math)
// was ~3x longer than flashinfer's oneshot layout (threads_per_token = hidden/VEC, cluster
// of >=128-thr blocks, single chunk per thread). At small T the kernel IS its latency chain.
constexpr int THREADS = 768;      // 24 warps = H/VEC: one bf16x8 chunk per thread
constexpr int VEC = 8;            // bf16x8 = 16B = one float4
constexpr int PER_THREAD = H / THREADS;        // 8
constexpr int CHUNKS = PER_THREAD / VEC;       // 1
constexpr int MAX_SLOTS = 1024;   // max decode tokens per launch (graph per-bs => static N per graph)
constexpr int MAX_RANKS = 8;
constexpr int MAX_TOPK = 8;
constexpr int NBUFS = 3;          // Lamport triple buffering (mod-3 rotation; 2 rounds of slack)
constexpr int MAX_NC = 4;         // v7: max gemm2 column chunks (matches csrc fe_chunk::MAX_NC)
constexpr uint32_t SENTINEL2 = 0x80008000u;  // two packed bf16 -0.0 = "not arrived"

static_assert(H % (THREADS * VEC) == 0, "H must tile evenly");

// --------------------------------------------------------------------------------- comm plumbing
// Per-rank symmetric buffers (allocated by the Python helper, exchanged via CUDA IPC):
//   data    : [NBUFS][MAX_SLOTS][MAX_RANKS][H]  bf16  peers PUSH their partial rows into my slots.
//             SEQ-FLAG path uses buffer 0 only (layout-compatible with the validated v1).
//             LAMPORT path rotates buf = counter[tok] % NBUFS and requires ALL THREE buffers
//             sentinel-initialized (0x8000 per bf16) by the Python side at init.
//   flags   : [MAX_SLOTS][MAX_RANKS]     u32    peers store their sequence value here (seq path only)
//   counter : [MAX_SLOTS]                u32    LOCAL per-slot invocation counter (never remote)
// MODE MIXING HAZARD: after any seq-path call, buffer 0 holds stale non-sentinel payloads; a
// later lamport call in the same process would consume them as arrived. Pick ONE mode per init.
struct CommView {
  __nv_bfloat16* data[MAX_RANKS];   // data[r] = rank r's data buffer base (r==me -> local VA)
  uint32_t* flags[MAX_RANKS];       // flags[r] = rank r's flag buffer base
  uint32_t* counter;                // my local counter buffer
  int rank;
  int world;
};

__device__ __forceinline__ uint32_t ld_acquire_sys(uint32_t const* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// ---- Lamport-variant primitives ----
// .relaxed.sys (not .volatile, not weak) on BOTH sides of the protocol, per the PTX memory model:
//   - single-copy atomicity (PTX ISA 8.10.3) holds per 32-bit ELEMENT of a v4.b32 (vector ops are
//     modelled as scalar-element ops in unspecified order, 8.2.4) and ONLY between morally-strong
//     ops (8.7: both strong, scopes include each other, same proxy, complete overlap). Weak ops
//     would be a data race -> no tearing guarantee at all.
//   - .volatile is semantically relaxed-at-sys-scope with extra implementation constraints; the
//     ISA (8.4, volatile operation) says to prefer ld/st.relaxed.sys for inter-thread sync perf.
// No acquire anywhere: the spin-read IS the data read (value dependency; nothing else to order).
__device__ __forceinline__ uint4 ld_relaxed_sys_v4(void const* p) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_sys_v4(void* p, uint4 v) {
  asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y),
               "r"(v.z), "r"(v.w)
               : "memory");
}
// Payload sanitization: any REAL bf16 -0.0 becomes +0.0 (numerically identical) so 0x8000 on the
// wire can only ever mean "not arrived". __vcmpeq2 = per-u16 compare, 0xFFFF mask per match.
__device__ __forceinline__ uint32_t flush_neg_zero2(uint32_t w) {
  return w & ~__vcmpeq2(w, SENTINEL2);
}
__device__ __forceinline__ bool has_sentinel(uint4 v) {
  return (__vcmpeq2(v.x, SENTINEL2) | __vcmpeq2(v.y, SENTINEL2) | __vcmpeq2(v.z, SENTINEL2) |
          __vcmpeq2(v.w, SENTINEL2)) != 0u;
}

template <int NT>  // v7: window blocks run NT = COLW/VEC threads (384 @ n_c=2), not THREADS
__device__ __forceinline__ float block_reduce_sum_t(float val, float* smem) {
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) val += __shfl_xor_sync(0xffffffff, val, m);
  if (lane == 0) smem[wid] = val;
  __syncthreads();
  if (wid == 0) {
    val = (threadIdx.x < (NT >> 5)) ? smem[threadIdx.x] : 0.0f;
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) val += __shfl_xor_sync(0xffffffff, val, m);
    if (lane == 0) smem[0] = val;
  }
  __syncthreads();
  return smem[0];
}

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
  return block_reduce_sum_t<THREADS>(val, smem);
}

// ------------------------------------------------------------------------------------- the core
// Push my (register-resident, fp32) partial row for token `tok`, sync, reduce all ranks' rows,
// then residual+RMSNorm and store. `acc` holds my partial (PER_THREAD fp32 per thread).
template <int NRANKS>
__device__ __forceinline__ void exchange_reduce_norm_store(
    float (&acc)[PER_THREAD], CommView cv, int tok,
    __nv_bfloat16 const* __restrict__ residual_in,  // [N,H]
    __nv_bfloat16 const* __restrict__ weight,       // [H]
    __nv_bfloat16* __restrict__ residual_out,       // [N,H]
    __nv_bfloat16* __restrict__ norm_out,           // [N,H]
    float eps, float weight_bias, float* smem) {
  const int tid = threadIdx.x;
  const size_t slot_off = ((size_t)tok * MAX_RANKS + cv.rank) * H;  // slot (tok, my_rank)

  // ---- (2) push my partial to every rank's slot (bf16, 128-bit stores) ----
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const int base = tid * VEC + c * (THREADS * VEC);
    __nv_bfloat16 pk[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) pk[j] = __float2bfloat16(acc[c * VEC + j]);
    float4 v = *reinterpret_cast<float4*>(pk);
#pragma unroll
    for (int r = 0; r < NRANKS; ++r)
      *reinterpret_cast<float4*>(cv.data[r] + slot_off + base) = v;
  }
  // Cross-device ordering: each thread's data stores must be visible to the peer before ANY flag
  // store. Per the PTX memory model this needs NO per-thread fence: bar.sync is a synchronizing
  // operation (ISA 8.4) that puts every thread's prior data stores in causality order before the
  // flag threads' st.release.sys, and a release PATTERN's coverage extends to other threads'
  // prior ops "through the transitive nature of causality order" (ISA 8.8, footnote 1). The
  // earlier per-thread __threadfence_system here was redundant (the TP2 prototype's actual bug
  // was .gpu SCOPE on the flags — moral strength, not a missing fence). Production precedent:
  // sglang/vLLM custom_all_reduce.cuh multi_gpu_barrier<_,_,need_fence=true> = exactly
  // __syncthreads(); st.release.sys; ld.acquire.sys spin.
  __syncthreads();

  // ---- (3)+(4) sequence flags: signal peers, spin locally ----
  const uint32_t seq = cv.counter[tok] + 1;  // same value in all threads (read-only here)
  if (tid < NRANKS) {
    st_release_sys(cv.flags[tid] + (size_t)tok * MAX_RANKS + cv.rank, seq);  // tid-th rank's flag
    uint32_t const* my_flag = cv.flags[cv.rank] + (size_t)tok * MAX_RANKS + tid;
    while (ld_acquire_sys(my_flag) != seq) { /* local spin */ }
  }
  __syncthreads();  // every rank's row for `tok` is now in MY local data slots

  // ---- (5)+(6) reduce + residual + rmsnorm ----
  float ssq = 0.0f;
  float out[PER_THREAD];
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const int base = tid * VEC + c * (THREADS * VEC);
    float f[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) f[j] = 0.0f;
#pragma unroll
    for (int r = 0; r < NRANKS; ++r) {
      float4 v = *reinterpret_cast<float4 const*>(cv.data[cv.rank] + ((size_t)tok * MAX_RANKS + r) * H + base);
      __nv_bfloat16* b = reinterpret_cast<__nv_bfloat16*>(&v);
#pragma unroll
      for (int j = 0; j < VEC; ++j) f[j] += __bfloat162float(b[j]);
    }
    float4 rv = *reinterpret_cast<float4 const*>(residual_in + (size_t)tok * H + base);
    __nv_bfloat16* rb = reinterpret_cast<__nv_bfloat16*>(&rv);
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      float o = f[j] + __bfloat162float(rb[j]);
      out[c * VEC + j] = o;
      ssq += o * o;
    }
  }
  const float rinv = rsqrtf(block_reduce_sum(ssq, smem) / float(H) + eps);

#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const int base = tid * VEC + c * (THREADS * VEC);
    float4 wv = *reinterpret_cast<float4 const*>(weight + base);
    __nv_bfloat16* wb = reinterpret_cast<__nv_bfloat16*>(&wv);
    __nv_bfloat16 res_pk[VEC], nrm_pk[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const float o = out[c * VEC + j];
      res_pk[j] = __float2bfloat16(o);
      nrm_pk[j] = __float2bfloat16(o * rinv * (weight_bias + __bfloat162float(wb[j])));
    }
    *reinterpret_cast<float4*>(residual_out + (size_t)tok * H + base) = *reinterpret_cast<float4*>(res_pk);
    *reinterpret_cast<float4*>(norm_out + (size_t)tok * H + base) = *reinterpret_cast<float4*>(nrm_pk);
  }

  // ---- (7) advance MY counter for this slot (replay-safe progression) ----
  // v4: no __syncthreads needed — every thread's `seq` read happened before the barrier at
  // step (4); the next kernel's read is stream/PDL-ordered behind this kernel's completion.
  if (tid == 0) cv.counter[tok] = seq;
}

// ----------------------------------------------------------------------- the core, LAMPORT variant
// Same contract as exchange_reduce_norm_store, but the DATA is its own arrival signal:
//   push : flush -0.0 payloads to +0.0, store bf16x8 into PEERS' current-buffer slots only
//          (my own contribution rides in registers as the SAME flushed bf16 wire bytes, so every
//          rank sums identical values in identical rank order -> bitwise-identical outputs).
//   spin : volatile-reload each 16B chunk of each peer row until no lane == 0x8000. No fence, no
//          flag, no __syncthreads before the spin — each thread consumes exactly the offsets its
//          peer-thread twin produced, and a 16B chunk is valid iff all its lanes are non-sentinel.
//   clear: after the reduce, re-sentinel the PREVIOUS buffer (buf-1 mod 3) for this token.
//          Safety of the write-write race (my clear vs a peer's future push into the same slots):
//          a peer writes buffer b again at invocation cnt+3; it can only get there after observing
//          my cnt+1 and cnt+2 pushes, and my cnt+1 push is issued a full kernel-boundary after
//          these clear stores committed to my L2 — the coherence point where its NVLink writes
//          land. Triple buffering buys exactly that kernel-boundary of slack (the production
//          TRT-LLM/flashinfer rotation); double buffering would race.
template <int NRANKS>
__device__ __forceinline__ void exchange_reduce_norm_store_lamport(
    float (&acc)[PER_THREAD], CommView cv, int tok,
    __nv_bfloat16 const* __restrict__ residual_in,  // [N,H]
    __nv_bfloat16 const* __restrict__ weight,       // [H]
    __nv_bfloat16* __restrict__ residual_out,       // [N,H]
    __nv_bfloat16* __restrict__ norm_out,           // [N,H]
    float eps, float weight_bias, float* smem) {
  const int tid = threadIdx.x;
  const uint32_t cnt = cv.counter[tok];
  const int buf = (int)(cnt % NBUFS);
  const int prev = (int)((cnt + NBUFS - 1) % NBUFS);
  const size_t buf_elems = (size_t)MAX_SLOTS * MAX_RANKS * H;

  // pack my partial as FLUSHED bf16 — the exact wire bytes every rank (incl. me) will consume
  uint4 mine[CHUNKS];
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    __nv_bfloat16 pk[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) pk[j] = __float2bfloat16(acc[c * VEC + j]);
    uint4 w = *reinterpret_cast<uint4*>(pk);
    w.x = flush_neg_zero2(w.x);
    w.y = flush_neg_zero2(w.y);
    w.z = flush_neg_zero2(w.z);
    w.w = flush_neg_zero2(w.w);
    mine[c] = w;
  }

  // ---- (2) push to PEERS' current-buffer slots (self-push skipped) ----
  const size_t slot_off = buf * buf_elems + ((size_t)tok * MAX_RANKS + cv.rank) * H;
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const int base = tid * VEC + c * (THREADS * VEC);
#pragma unroll
    for (int r = 0; r < NRANKS; ++r) {
      if (r == cv.rank) continue;
      st_relaxed_sys_v4(cv.data[r] + slot_off + base, mine[c]);
    }
  }

  // ---- (4)+(5) spin-consume peer rows locally, fp32-accumulate in ascending rank order ----
  // v4 (2026-07-02, NCU-driven): the old per-peer `do { ld } while (sentinel)` loop made peer
  // r+1's load wait on peer r's ARRIVAL — 3x serialized load latency per thread. The stall-
  // annotated SASS put the top samples on the sentinel-check LOP3 (0x80008000) consuming each
  // load. Now ALL peer loads issue back-to-back and the SET is polled until every chunk is
  // sentinel-free; accumulation stays in ASCENDING RANK ORDER from registers (all TP replicas
  // must produce bit-identical activations or they drift apart downstream).
  float ssq = 0.0f;
  float out[PER_THREAD];
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const int base = tid * VEC + c * (THREADS * VEC);
    uint4 pv[NRANKS];
#pragma unroll
    for (int r = 0; r < NRANKS; ++r) {  // issue all concurrently (no control dependency)
      if (r == cv.rank) continue;
      pv[r] = ld_relaxed_sys_v4(cv.data[cv.rank] + buf * buf_elems +
                                ((size_t)tok * MAX_RANKS + r) * H + base);
    }
    bool pending = true;
    while (pending) {  // re-load only not-yet-arrived chunks; exit on a clean sweep
      pending = false;
#pragma unroll
      for (int r = 0; r < NRANKS; ++r) {
        if (r == cv.rank) continue;
        if (has_sentinel(pv[r])) {
          pv[r] = ld_relaxed_sys_v4(cv.data[cv.rank] + buf * buf_elems +
                                    ((size_t)tok * MAX_RANKS + r) * H + base);
          pending = true;
        }
      }
    }
    pv[cv.rank] = mine[c];
    float f[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) f[j] = 0.0f;
#pragma unroll
    for (int r = 0; r < NRANKS; ++r) {  // deterministic rank-order accumulate
      __nv_bfloat16 const* b = reinterpret_cast<__nv_bfloat16 const*>(&pv[r]);
#pragma unroll
      for (int j = 0; j < VEC; ++j) f[j] += __bfloat162float(b[j]);
    }
    float4 rv = *reinterpret_cast<float4 const*>(residual_in + (size_t)tok * H + base);
    __nv_bfloat16* rb = reinterpret_cast<__nv_bfloat16*>(&rv);
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      float o = f[j] + __bfloat162float(rb[j]);
      out[c * VEC + j] = o;
      ssq += o * o;
    }
  }
  const float rinv = rsqrtf(block_reduce_sum(ssq, smem) / float(H) + eps);

#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const int base = tid * VEC + c * (THREADS * VEC);
    float4 wv = *reinterpret_cast<float4 const*>(weight + base);
    __nv_bfloat16* wb = reinterpret_cast<__nv_bfloat16*>(&wv);
    __nv_bfloat16 res_pk[VEC], nrm_pk[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const float o = out[c * VEC + j];
      res_pk[j] = __float2bfloat16(o);
      nrm_pk[j] = __float2bfloat16(o * rinv * (weight_bias + __bfloat162float(wb[j])));
    }
    *reinterpret_cast<float4*>(residual_out + (size_t)tok * H + base) = *reinterpret_cast<float4*>(res_pk);
    *reinterpret_cast<float4*>(norm_out + (size_t)tok * H + base) = *reinterpret_cast<float4*>(nrm_pk);
  }

  // ---- (7) re-sentinel the PREVIOUS buffer for this token; advance the counter ----
  // v4: no __syncthreads here. The clear targets prev (never read by this kernel: spin reads
  // buf, prev != buf at NBUFS=3); every thread's counter READ is already ordered before this
  // write by the block_reduce_sum barriers; the next kernel's read is stream/PDL-ordered.
  const size_t prev_row = prev * buf_elems + (size_t)tok * MAX_RANKS * H;
  const uint4 sent = make_uint4(SENTINEL2, SENTINEL2, SENTINEL2, SENTINEL2);
  for (int i = tid * VEC; i < NRANKS * H; i += THREADS * VEC)
    st_relaxed_sys_v4(cv.data[cv.rank] + prev_row + i, sent);
  // wrap rule keeps the %3 cycle intact (2^32 % 3 != 0, so a plain wrap would repeat a buffer:
  // ...0xFFFFFFFE(%3=2) -> 0xFFFFFFFF(%3=0) -> 0(%3=0) = STALE). Skip 0xFFFFFFFF instead.
  if (tid == 0) cv.counter[tok] = (cnt == 0xFFFFFFFEu) ? 0u : cnt + 1u;
}

// ------------------------------------------------------------- the core, LAMPORT TWO-SHOT (v6)
// For T ≳ 192 the one-shot push loses to stock's custom AR: bytes scale as 3×H per token
// (measured: 32.9µs vs stock 31.1 @256, 94µs vs ~flat @1024). Two-shot halves the wire bytes
// (RS + AG = 1.5×H) at the cost of a second dependency round — the TRT-LLM twoshot pattern on
// our buffer geometry. Dispatch: one-shot ≤ threshold, two-shot above (host/python decides).
//
// Segment ownership: rank o owns row chunks [o*SEGC, (o+1)*SEGC) (SEGC = H/VEC/NRANKS).
// Thread t (768 = one/chunk) has seg = t/SEGC, off natural chunk offset t.
//   P1 (reduce-scatter): thread t pushes its FLUSHED acc chunk to OWNER seg's buffer at cell
//       (row = me, chunk t). Owner-side threads (seg == me) v4-concurrent-poll the 3 peer rows
//       at chunk t, then reduce in ascending rank order (fp32) + own reg contribution.
//   P1.5: owner threads FLUSH the reduced chunk (it goes on the wire — a -0.0 sum would read
//       as a sentinel) and push it to all peers' cells (row me, chunk t). Owner threads keep
//       the (flushed) value in registers — no local store needed: the 768-thread full-row
//       mapping means owner threads ARE the norm-phase holders of their own segment.
//   P2 (all-gather): threads with seg s != me poll cell (row s, chunk t) until the owner's
//       reduced chunk lands.
//   Norm: every thread now holds its full-row chunk (own seg: reg; others: P2) → residual add,
//       ssq, block reduce, store. BIT-IDENTITY BY CONSTRUCTION: each segment is summed exactly
//       once (by its owner) and broadcast — all ranks see identical bytes, no ordering caveat.
// Cell-collision proof (rank p's buffer): P1 writes (m≠p, seg p); P2 writes (o≠p, seg o).
// Overlap needs m==o AND seg p==seg o → p==o, excluded. Disjoint. Same NBUFS rotation/clear/
// counter as one-shot (all cells live in the same (buf, tok) region; causality proof carries).
template <int NRANKS>
__device__ __forceinline__ void exchange_reduce_norm_store_lamport2(
    float (&acc)[PER_THREAD], CommView cv, int tok,
    __nv_bfloat16 const* __restrict__ residual_in, __nv_bfloat16 const* __restrict__ weight,
    __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ norm_out,
    float eps, float weight_bias, float* smem) {
  static_assert(CHUNKS == 1, "two-shot assumes the v3 one-chunk-per-thread layout");
  constexpr int SEGC = (H / VEC) / NRANKS;  // chunks per owned segment (192 @ TP4)
  const int tid = threadIdx.x;
  const int seg = tid / SEGC;  // which rank owns MY chunk
  const uint32_t cnt = cv.counter[tok];
  const int buf = (int)(cnt % NBUFS);
  const int prev = (int)((cnt + NBUFS - 1) % NBUFS);
  const size_t buf_elems = (size_t)MAX_SLOTS * MAX_RANKS * H;
  const size_t row_base = buf * buf_elems + (size_t)tok * MAX_RANKS * H;  // + row*H + elem
  const int base = tid * VEC;  // my chunk's element offset in the row

  // pack + flush my partial chunk (wire bytes)
  uint4 mine;
  {
    __nv_bfloat16 pk[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) pk[j] = __float2bfloat16(acc[j]);
    uint4 w = *reinterpret_cast<uint4*>(pk);
    w.x = flush_neg_zero2(w.x); w.y = flush_neg_zero2(w.y);
    w.z = flush_neg_zero2(w.z); w.w = flush_neg_zero2(w.w);
    mine = w;
  }

  // ---- P1 push: my partial chunk -> owner's buffer, cell (row me, chunk t) ----
  if (seg != cv.rank)
    st_relaxed_sys_v4(cv.data[seg] + row_base + (size_t)cv.rank * H + base, mine);

  float red[VEC];  // owner threads: the reduced segment chunk (fp32)
  if (seg == cv.rank) {
    // ---- P1 reduce (owner): v4-concurrent-poll peers' rows at my chunk ----
    uint4 pv[NRANKS];
#pragma unroll
    for (int r = 0; r < NRANKS; ++r) {
      if (r == cv.rank) continue;
      pv[r] = ld_relaxed_sys_v4(cv.data[cv.rank] + row_base + (size_t)r * H + base);
    }
    bool pending = true;
    while (pending) {
      pending = false;
#pragma unroll
      for (int r = 0; r < NRANKS; ++r) {
        if (r == cv.rank) continue;
        if (has_sentinel(pv[r])) {
          pv[r] = ld_relaxed_sys_v4(cv.data[cv.rank] + row_base + (size_t)r * H + base);
          pending = true;
        }
      }
    }
    pv[cv.rank] = mine;
#pragma unroll
    for (int j = 0; j < VEC; ++j) red[j] = 0.0f;
#pragma unroll
    for (int r = 0; r < NRANKS; ++r) {
      __nv_bfloat16 const* b = reinterpret_cast<__nv_bfloat16 const*>(&pv[r]);
#pragma unroll
      for (int j = 0; j < VEC; ++j) red[j] += __bfloat162float(b[j]);
    }
    // ---- P1.5: flush + broadcast the reduced chunk to cell (row me, chunk t) everywhere ----
    __nv_bfloat16 rk[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) rk[j] = __float2bfloat16(red[j]);
    uint4 rw = *reinterpret_cast<uint4*>(rk);
    rw.x = flush_neg_zero2(rw.x); rw.y = flush_neg_zero2(rw.y);
    rw.z = flush_neg_zero2(rw.z); rw.w = flush_neg_zero2(rw.w);
#pragma unroll
    for (int r = 0; r < NRANKS; ++r) {
      if (r == cv.rank) continue;
      st_relaxed_sys_v4(cv.data[r] + row_base + (size_t)cv.rank * H + base, rw);
    }
    // re-read as bf16 for the norm path so all ranks use the IDENTICAL wire bytes
    __nv_bfloat16 const* rb = reinterpret_cast<__nv_bfloat16 const*>(&rw);
#pragma unroll
    for (int j = 0; j < VEC; ++j) red[j] = __bfloat162float(rb[j]);
  } else {
    // ---- P2: poll the owner's reduced chunk at cell (row seg, chunk t) in MY buffer ----
    void const* p = cv.data[cv.rank] + row_base + (size_t)seg * H + base;
    uint4 v = ld_relaxed_sys_v4(p);
    while (has_sentinel(v)) v = ld_relaxed_sys_v4(p);
    __nv_bfloat16 const* b = reinterpret_cast<__nv_bfloat16 const*>(&v);
#pragma unroll
    for (int j = 0; j < VEC; ++j) red[j] = __bfloat162float(b[j]);
  }

  // ---- residual + rmsnorm on the full row (every thread holds its chunk) ----
  float ssq = 0.0f;
  float out[VEC];
  {
    float4 rv = *reinterpret_cast<float4 const*>(residual_in + (size_t)tok * H + base);
    __nv_bfloat16* rb = reinterpret_cast<__nv_bfloat16*>(&rv);
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      float o = red[j] + __bfloat162float(rb[j]);
      out[j] = o;
      ssq += o * o;
    }
  }
  const float rinv = rsqrtf(block_reduce_sum(ssq, smem) / float(H) + eps);
  {
    float4 wv = *reinterpret_cast<float4 const*>(weight + base);
    __nv_bfloat16* wb = reinterpret_cast<__nv_bfloat16*>(&wv);
    __nv_bfloat16 res_pk[VEC], nrm_pk[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      res_pk[j] = __float2bfloat16(out[j]);
      nrm_pk[j] = __float2bfloat16(out[j] * rinv * (weight_bias + __bfloat162float(wb[j])));
    }
    *reinterpret_cast<float4*>(residual_out + (size_t)tok * H + base) = *reinterpret_cast<float4*>(res_pk);
    *reinterpret_cast<float4*>(norm_out + (size_t)tok * H + base) = *reinterpret_cast<float4*>(nrm_pk);
  }

  // ---- re-sentinel prev buffer + advance counter (same rules as one-shot) ----
  const size_t prev_row = prev * buf_elems + (size_t)tok * MAX_RANKS * H;
  const uint4 sent = make_uint4(SENTINEL2, SENTINEL2, SENTINEL2, SENTINEL2);
  for (int i = tid * VEC; i < NRANKS * H; i += THREADS * VEC)
    st_relaxed_sys_v4(cv.data[cv.rank] + prev_row + i, sent);
  if (tid == 0) cv.counter[tok] = (cnt == 0xFFFFFFFEu) ? 0u : cnt + 1u;
}

// ----------------------------------------------------- v7: COLUMN-WINDOW consumers (rung-3 S2)
// The gemm2 producer is column-chunked (csrc fe_chunk patch, n_c planes packed at chunk width);
// consumer k processes global columns [col_off, col_off+COLW) as soon as chunk k's GEMM event
// fires — overlapping chunk k's epilogue with chunk k+1's GEMM. Contract deltas vs the
// full-row core (07 plan §5 R1 / §7 S2):
//   - ALL chunk consumers of one logical call share the SAME Lamport (buf, tok) slot: each
//     reads counter[tok] (identical value — consumers are stream-ordered within the call) and
//     only the LAST chunk's kernel advances it (advance-flag). Column regions are disjoint, so
//     the v6 cell-collision and clear-race proofs carry per-region.
//   - prev-buffer sentinel-clear is SPLIT: consumer k re-sentinels only its column window
//     (all NRANKS rows). Union over chunks == the v6 full clear; completes before the counter
//     advance in the last chunk (same kernel-boundary slack argument as v6).
//   - RMSNorm needs the full-row ssq: consumer k<last block-reduces its window ssq into a
//     LOCAL fp32 slot ssq_slots[tok*MAX_NC+k] and writes residual_out (bf16(o)) for its
//     window ONLY. The LAST consumer adds the stored partials (identical order on every rank
//     -> identical rinv -> no cross-rank drift), then writes norm_out for the WHOLE row: its
//     own window from fp32 registers, earlier windows by re-reading the bf16 o from
//     residual_out (same stream -> coherent; the bf16 re-read is an intermediate rounding the
//     exact-emulation ref models; within the 2-ulp bar).
//   - Lamport modes only (1 = one-shot, 2 = two-shot). The seq-flag path is not windowed.
// Thread layout: NTW = COLW/VEC threads, ONE bf16x8 chunk per thread (v3 law: at decode T the
// kernel IS its latency chain). Two-shot segment ownership is defined WITHIN the window:
// rank o owns window-chunks [o*SEGW, (o+1)*SEGW) — ownership differs from the full-row v6
// mapping but each column is still summed exactly once in ascending rank order, so the
// reduced wire bytes are IDENTICAL to v6 two-shot.

template <int NRANKS, int COLW>
__device__ __forceinline__ void exchange_win_lamport1(
    float (&acc)[VEC], CommView cv, int tok, int col_off, uint32_t cnt, float (&red)[VEC]) {
  const int tid = threadIdx.x;
  const int col = col_off + tid * VEC;  // my global column base
  const int buf = (int)(cnt % NBUFS);
  const size_t buf_elems = (size_t)MAX_SLOTS * MAX_RANKS * H;
  const size_t row_base = buf * buf_elems + (size_t)tok * MAX_RANKS * H;

  // pack + flush my partial chunk (wire bytes; self rides in registers as the same bytes)
  uint4 mine;
  {
    __nv_bfloat16 pk[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) pk[j] = __float2bfloat16(acc[j]);
    uint4 w = *reinterpret_cast<uint4*>(pk);
    w.x = flush_neg_zero2(w.x); w.y = flush_neg_zero2(w.y);
    w.z = flush_neg_zero2(w.z); w.w = flush_neg_zero2(w.w);
    mine = w;
  }
  // push to PEERS' slots (row = me, my window columns only)
#pragma unroll
  for (int r = 0; r < NRANKS; ++r) {
    if (r == cv.rank) continue;
    st_relaxed_sys_v4(cv.data[r] + row_base + (size_t)cv.rank * H + col, mine);
  }
  // v4-concurrent-poll peers' rows at my columns (v4 lesson: issue all, re-load laggards)
  uint4 pv[NRANKS];
#pragma unroll
  for (int r = 0; r < NRANKS; ++r) {
    if (r == cv.rank) continue;
    pv[r] = ld_relaxed_sys_v4(cv.data[cv.rank] + row_base + (size_t)r * H + col);
  }
  bool pending = true;
  while (pending) {
    pending = false;
#pragma unroll
    for (int r = 0; r < NRANKS; ++r) {
      if (r == cv.rank) continue;
      if (has_sentinel(pv[r])) {
        pv[r] = ld_relaxed_sys_v4(cv.data[cv.rank] + row_base + (size_t)r * H + col);
        pending = true;
      }
    }
  }
  pv[cv.rank] = mine;
#pragma unroll
  for (int j = 0; j < VEC; ++j) red[j] = 0.0f;
#pragma unroll
  for (int r = 0; r < NRANKS; ++r) {  // deterministic ascending-rank fp32 accumulate
    __nv_bfloat16 const* b = reinterpret_cast<__nv_bfloat16 const*>(&pv[r]);
#pragma unroll
    for (int j = 0; j < VEC; ++j) red[j] += __bfloat162float(b[j]);
  }
}

template <int NRANKS, int COLW>
__device__ __forceinline__ void exchange_win_lamport2(
    float (&acc)[VEC], CommView cv, int tok, int col_off, uint32_t cnt, float (&red)[VEC]) {
  constexpr int NTW = COLW / VEC;
  constexpr int SEGW = NTW / NRANKS;  // window chunks per owner (96 @ TP4, COLW=3072)
  static_assert(NTW % NRANKS == 0, "window must split evenly across ranks");
  const int tid = threadIdx.x;
  const int seg = tid / SEGW;         // which rank owns MY window chunk
  const int col = col_off + tid * VEC;
  const int buf = (int)(cnt % NBUFS);
  const size_t buf_elems = (size_t)MAX_SLOTS * MAX_RANKS * H;
  const size_t row_base = buf * buf_elems + (size_t)tok * MAX_RANKS * H;

  uint4 mine;
  {
    __nv_bfloat16 pk[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) pk[j] = __float2bfloat16(acc[j]);
    uint4 w = *reinterpret_cast<uint4*>(pk);
    w.x = flush_neg_zero2(w.x); w.y = flush_neg_zero2(w.y);
    w.z = flush_neg_zero2(w.z); w.w = flush_neg_zero2(w.w);
    mine = w;
  }

  // ---- P1 push: my partial chunk -> OWNER's buffer, cell (row me, col) ----
  if (seg != cv.rank)
    st_relaxed_sys_v4(cv.data[seg] + row_base + (size_t)cv.rank * H + col, mine);

  if (seg == cv.rank) {
    // ---- P1 reduce (owner): concurrent-poll peers' rows at my columns ----
    uint4 pv[NRANKS];
#pragma unroll
    for (int r = 0; r < NRANKS; ++r) {
      if (r == cv.rank) continue;
      pv[r] = ld_relaxed_sys_v4(cv.data[cv.rank] + row_base + (size_t)r * H + col);
    }
    bool pending = true;
    while (pending) {
      pending = false;
#pragma unroll
      for (int r = 0; r < NRANKS; ++r) {
        if (r == cv.rank) continue;
        if (has_sentinel(pv[r])) {
          pv[r] = ld_relaxed_sys_v4(cv.data[cv.rank] + row_base + (size_t)r * H + col);
          pending = true;
        }
      }
    }
    pv[cv.rank] = mine;
    float redf[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) redf[j] = 0.0f;
#pragma unroll
    for (int r = 0; r < NRANKS; ++r) {  // ascending rank order — identical bytes to v6 two-shot
      __nv_bfloat16 const* b = reinterpret_cast<__nv_bfloat16 const*>(&pv[r]);
#pragma unroll
      for (int j = 0; j < VEC; ++j) redf[j] += __bfloat162float(b[j]);
    }
    // ---- P1.5: flush + broadcast the reduced chunk to all peers at (row me, col) ----
    __nv_bfloat16 rk[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) rk[j] = __float2bfloat16(redf[j]);
    uint4 rw = *reinterpret_cast<uint4*>(rk);
    rw.x = flush_neg_zero2(rw.x); rw.y = flush_neg_zero2(rw.y);
    rw.z = flush_neg_zero2(rw.z); rw.w = flush_neg_zero2(rw.w);
#pragma unroll
    for (int r = 0; r < NRANKS; ++r) {
      if (r == cv.rank) continue;
      st_relaxed_sys_v4(cv.data[r] + row_base + (size_t)cv.rank * H + col, rw);
    }
    __nv_bfloat16 const* rb = reinterpret_cast<__nv_bfloat16 const*>(&rw);
#pragma unroll
    for (int j = 0; j < VEC; ++j) red[j] = __bfloat162float(rb[j]);  // identical wire bytes
  } else {
    // ---- P2: poll the owner's reduced chunk at cell (row seg, col) in MY buffer ----
    void const* p = cv.data[cv.rank] + row_base + (size_t)seg * H + col;
    uint4 v = ld_relaxed_sys_v4(p);
    while (has_sentinel(v)) v = ld_relaxed_sys_v4(p);
    __nv_bfloat16 const* b = reinterpret_cast<__nv_bfloat16 const*>(&v);
#pragma unroll
    for (int j = 0; j < VEC; ++j) red[j] = __bfloat162float(b[j]);
  }
}

// Shared window tail: residual add, window residual_out, partial-ssq vs last-chunk finish
// (rinv + FULL-row norm), SPLIT prev-buffer clear, advance-flag counter update.
template <int NRANKS, int COLW>
__device__ __forceinline__ void window_finish(
    float (&red)[VEC], CommView cv, int tok, int col_off, uint32_t cnt,
    __nv_bfloat16 const* __restrict__ residual_in,  // [N,H]
    __nv_bfloat16 const* __restrict__ weight,       // [H]
    __nv_bfloat16* __restrict__ residual_out,       // [N,H]
    __nv_bfloat16* __restrict__ norm_out,           // [N,H]
    float eps, float weight_bias, float* smem,
    float* __restrict__ ssq_slots,  // [MAX_SLOTS, MAX_NC] fp32, LOCAL (identical on all ranks)
    int chunk_idx, int n_c, bool is_last) {
  constexpr int NTW = COLW / VEC;
  const int tid = threadIdx.x;
  const int col = col_off + tid * VEC;

  // residual add + my-window partial ssq
  float o[VEC];
  float ssq = 0.0f;
  {
    float4 rv = *reinterpret_cast<float4 const*>(residual_in + (size_t)tok * H + col);
    __nv_bfloat16* rb = reinterpret_cast<__nv_bfloat16*>(&rv);
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      o[j] = red[j] + __bfloat162float(rb[j]);
      ssq += o[j] * o[j];
    }
  }
  // residual_out for my window — bf16(o), the exact bytes the last chunk re-reads for norm
  {
    __nv_bfloat16 res_pk[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) res_pk[j] = __float2bfloat16(o[j]);
    *reinterpret_cast<float4*>(residual_out + (size_t)tok * H + col) =
        *reinterpret_cast<float4 const*>(res_pk);
  }

  const float ssq_win = block_reduce_sum_t<NTW>(ssq, smem);

  if (!is_last) {
    if (tid == 0) ssq_slots[(size_t)tok * MAX_NC + chunk_idx] = ssq_win;
  } else {
    // total ssq: my window + stored partials, SAME summation order on every thread and rank
    float total = ssq_win;
    for (int c = 0; c < n_c; ++c)
      if (c != chunk_idx) total += ssq_slots[(size_t)tok * MAX_NC + c];
    const float rinv = rsqrtf(total / float(H) + eps);
    // my window: norm from fp32 registers (v6 numerics)
    {
      float4 wv = *reinterpret_cast<float4 const*>(weight + col);
      __nv_bfloat16* wb = reinterpret_cast<__nv_bfloat16*>(&wv);
      __nv_bfloat16 nrm_pk[VEC];
#pragma unroll
      for (int j = 0; j < VEC; ++j)
        nrm_pk[j] = __float2bfloat16(o[j] * rinv * (weight_bias + __bfloat162float(wb[j])));
      *reinterpret_cast<float4*>(norm_out + (size_t)tok * H + col) =
          *reinterpret_cast<float4 const*>(nrm_pk);
    }
    // earlier windows: re-read bf16 o from residual_out (stream-ordered writes by the
    // earlier chunk consumers), scale, store norm_out
    for (int base = tid * VEC; base < H; base += NTW * VEC) {
      if (base >= col_off && base < col_off + COLW) continue;  // mine, already written
      float4 ov = *reinterpret_cast<float4 const*>(residual_out + (size_t)tok * H + base);
      float4 wv = *reinterpret_cast<float4 const*>(weight + base);
      __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);
      __nv_bfloat16* wb = reinterpret_cast<__nv_bfloat16*>(&wv);
      __nv_bfloat16 nrm_pk[VEC];
#pragma unroll
      for (int j = 0; j < VEC; ++j)
        nrm_pk[j] = __float2bfloat16(__bfloat162float(ob[j]) * rinv *
                                     (weight_bias + __bfloat162float(wb[j])));
      *reinterpret_cast<float4*>(norm_out + (size_t)tok * H + base) =
          *reinterpret_cast<float4 const*>(nrm_pk);
    }
  }

  // SPLIT clear: re-sentinel MY window columns of the PREV buffer, all NRANKS rows.
  // Union over the n_c consumers == the v6 full clear; prev != buf at NBUFS=3, and the
  // write-write race safety vs a peer's future push carries per-region (v6 proof).
  const int prev = (int)((cnt + NBUFS - 1) % NBUFS);
  const size_t buf_elems = (size_t)MAX_SLOTS * MAX_RANKS * H;
  const size_t prev_base = prev * buf_elems + (size_t)tok * MAX_RANKS * H;
  const uint4 sent = make_uint4(SENTINEL2, SENTINEL2, SENTINEL2, SENTINEL2);
#pragma unroll
  for (int r = 0; r < NRANKS; ++r)  // NTW*VEC == COLW: one v4 store per rank covers my window
    st_relaxed_sys_v4(cv.data[cv.rank] + prev_base + (size_t)r * H + col, sent);

  // advance-flag: ONLY the last chunk's kernel advances the per-slot counter (wrap rule = v6)
  if (is_last && tid == 0) cv.counter[tok] = (cnt == 0xFFFFFFFEu) ? 0u : cnt + 1u;
}

// -------------------------------------------------------------------------------- entry kernels
// PDL (v5, template flag): launched via cudaLaunchKernelEx with programmatic stream
// serialization, the kernel may start BEFORE its predecessor completes — overlapping launch
// latency with the predecessor's tail (the in-model lever flashinfer ships with). Everything
// must wait for cudaGridDependencySynchronize(): inputs come from the predecessor AND the
// round counter is written at the END of the previous epilogue call (reading it early would
// rotate to the wrong Lamport buffer). No explicit trigger — implicit at kernel completion.
__device__ __forceinline__ void pdl_grid_sync() {
#if __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
}

template <int NRANKS, int MODE, bool PDL>  // MODE: 0=seq-flag, 1=lamport one-shot, 2=lamport two-shot
__global__ void ar_add_rmsnorm_kernel(
    __nv_bfloat16 const* __restrict__ partial_in,  // [N,H] my rank's pre-AR partial
    __nv_bfloat16 const* __restrict__ residual_in, __nv_bfloat16 const* __restrict__ weight,
    __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ norm_out,
    CommView cv, float eps, float weight_bias, int N) {
  __shared__ float smem[(THREADS + 31) / 32];  // one slot per warp (24 at THREADS=768)
  const int tok = blockIdx.x;
  if (PDL) pdl_grid_sync();
  if (tok >= N) return;
  float acc[PER_THREAD];
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const int base = threadIdx.x * VEC + c * (THREADS * VEC);
    float4 v = *reinterpret_cast<float4 const*>(partial_in + (size_t)tok * H + base);
    __nv_bfloat16* b = reinterpret_cast<__nv_bfloat16*>(&v);
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc[c * VEC + j] = __bfloat162float(b[j]);
  }
  if (MODE == 2)
    exchange_reduce_norm_store_lamport2<NRANKS>(acc, cv, tok, residual_in, weight, residual_out,
                                                norm_out, eps, weight_bias, smem);
  else if (MODE == 1)
    exchange_reduce_norm_store_lamport<NRANKS>(acc, cv, tok, residual_in, weight, residual_out,
                                               norm_out, eps, weight_bias, smem);
  else
    exchange_reduce_norm_store<NRANKS>(acc, cv, tok, residual_in, weight, residual_out, norm_out,
                                       eps, weight_bias, smem);
}

template <int NRANKS, int TOPK, int MODE, bool PDL>  // MODE: 0=seq, 1=lamport 1shot, 2=lamport 2shot
__global__ void moe_finalize_ar_add_rmsnorm_kernel(
    __nv_bfloat16 const* __restrict__ gemm2_out,   // [N*TOPK, H] permuted expert rows
    int32_t const* __restrict__ idx,               // token->permuted-row map; layout per idx_kmajor
    float const* __restrict__ scales,              // [N, TOPK] routing weights (fp32, t-major both ways)
    __nv_bfloat16 const* __restrict__ shared_out,  // [N, H] shared-expert partial, or nullptr
    __nv_bfloat16 const* __restrict__ residual_in, __nv_bfloat16 const* __restrict__ weight,
    __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ norm_out,
    CommView cv, float routed_scaling, float eps, float weight_bias, int idx_kmajor,
    int exp_rows, int N) {
  // exp_rows = the EXPORT's token count (k-major stride + row-bounds base). N = tokens to
  // process this launch — may be SMALLER than exp_rows (the deferred-AR consumer head-trims
  // CUDA-graph batch padding: e.g. consume T=496 of a 512-row moe export).
  __shared__ float smem[(THREADS + 31) / 32];  // one slot per warp (24 at THREADS=768)
  __shared__ int s_idx[TOPK];
  __shared__ float s_scl[TOPK];
  const int tok = blockIdx.x;
  if (PDL) pdl_grid_sync();
  if (tok >= N) return;
  if (threadIdx.x < TOPK) {
    // idx_kmajor=0: sglang jit convention idx[t*K + k]. idx_kmajor=1: TRT-LLM
    // unpermuted_row_to_permuted_row convention idx[k*exp_rows + t] (fe_export deep seam).
    s_idx[threadIdx.x] = idx[idx_kmajor ? ((size_t)threadIdx.x * exp_rows + tok)
                                        : ((size_t)tok * TOPK + threadIdx.x)];
    s_scl[threadIdx.x] = scales[(size_t)tok * TOPK + threadIdx.x];
  }
  __syncthreads();

  // ---- (1) finalize prologue: my partial = routed_scaling * sum_k scl_k * row_k (+ shared) ----
  float acc[PER_THREAD];
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const int base = threadIdx.x * VEC + c * (THREADS * VEC);
    float f[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) f[j] = 0.0f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      const int row = s_idx[k];
      if (row < 0 || row >= exp_rows * TOPK) continue;  // dropped / non-local (trtllm guard)
      float4 v = *reinterpret_cast<float4 const*>(gemm2_out + (size_t)row * H + base);
      __nv_bfloat16* b = reinterpret_cast<__nv_bfloat16*>(&v);
      const float s = s_scl[k];
#pragma unroll
      for (int j = 0; j < VEC; ++j) f[j] += s * __bfloat162float(b[j]);
    }
    if (shared_out != nullptr) {
      float4 sv = *reinterpret_cast<float4 const*>(shared_out + (size_t)tok * H + base);
      __nv_bfloat16* sb = reinterpret_cast<__nv_bfloat16*>(&sv);
#pragma unroll
      for (int j = 0; j < VEC; ++j) f[j] = f[j] * routed_scaling + __bfloat162float(sb[j]);
    } else {
#pragma unroll
      for (int j = 0; j < VEC; ++j) f[j] *= routed_scaling;
    }
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc[c * VEC + j] = f[j];
  }
  if (MODE == 2)
    exchange_reduce_norm_store_lamport2<NRANKS>(acc, cv, tok, residual_in, weight, residual_out,
                                                norm_out, eps, weight_bias, smem);
  else if (MODE == 1)
    exchange_reduce_norm_store_lamport<NRANKS>(acc, cv, tok, residual_in, weight, residual_out,
                                               norm_out, eps, weight_bias, smem);
  else
    exchange_reduce_norm_store<NRANKS>(acc, cv, tok, residual_in, weight, residual_out, norm_out,
                                       eps, weight_bias, smem);
}

// v7 deep-seam WINDOW entry: consume ONE column chunk of a fe_chunk'd gemm2 export.
// gemm2_out = the chunk's PACKED D plane (row stride g2_ld == chunk width, plane column 0 ==
// global column col_off). idx/scales = the same k-major TRT-LLM export as the full-row entry.
// Launch: N blocks x (COLW/VEC) threads.
template <int NRANKS, int TOPK, int MODE, bool PDL, int COLW>
__global__ void moe_finalize_ar_add_rmsnorm_win_kernel(
    __nv_bfloat16 const* __restrict__ gemm2_out,   // [exp_rows*TOPK, g2_ld] packed plane
    int64_t g2_ld,                                 // plane row stride (elements)
    int32_t const* __restrict__ idx,               // token->permuted-row map (per idx_kmajor)
    float const* __restrict__ scales,              // [exp_rows, TOPK] fp32 (scaling pre-folded)
    __nv_bfloat16 const* __restrict__ residual_in, __nv_bfloat16 const* __restrict__ weight,
    __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ norm_out,
    CommView cv, float eps, float weight_bias, int idx_kmajor, int exp_rows,
    int col_off, float* __restrict__ ssq_slots, int chunk_idx, int n_c, int is_last, int N) {
  constexpr int NTW = COLW / VEC;
  static_assert(MODE == 1 || MODE == 2, "window consumers are Lamport-only");
  __shared__ float smem[(NTW + 31) / 32];
  __shared__ int s_idx[TOPK];
  __shared__ float s_scl[TOPK];
  const int tok = blockIdx.x;
  if (PDL) pdl_grid_sync();  // gemm2 chunk + earlier consumers must complete (data deps)
  if (tok >= N) return;
  if (threadIdx.x < TOPK) {
    s_idx[threadIdx.x] = idx[idx_kmajor ? ((size_t)threadIdx.x * exp_rows + tok)
                                        : ((size_t)tok * TOPK + threadIdx.x)];
    s_scl[threadIdx.x] = scales[(size_t)tok * TOPK + threadIdx.x];
  }
  __syncthreads();
  const uint32_t cnt = cv.counter[tok];  // shared slot: read-only here; last chunk advances

  // finalize prologue over MY plane columns (plane col = global col - col_off = tid*VEC)
  float acc[VEC];
#pragma unroll
  for (int j = 0; j < VEC; ++j) acc[j] = 0.0f;
  const int pcol = threadIdx.x * VEC;
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    const int row = s_idx[k];
    if (row < 0 || row >= exp_rows * TOPK) continue;  // dropped / non-local (trtllm guard)
    float4 v = *reinterpret_cast<float4 const*>(gemm2_out + (size_t)row * g2_ld + pcol);
    __nv_bfloat16* b = reinterpret_cast<__nv_bfloat16*>(&v);
    const float s = s_scl[k];
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc[j] += s * __bfloat162float(b[j]);
  }

  float red[VEC];
  if (MODE == 2)
    exchange_win_lamport2<NRANKS, COLW>(acc, cv, tok, col_off, cnt, red);
  else
    exchange_win_lamport1<NRANKS, COLW>(acc, cv, tok, col_off, cnt, red);
  window_finish<NRANKS, COLW>(red, cv, tok, col_off, cnt, residual_in, weight, residual_out,
                              norm_out, eps, weight_bias, smem, ssq_slots, chunk_idx, n_c,
                              is_last != 0);
}

// ------------------------------------------------------------------------------------ host side
static CommView make_view(torch::Tensor data_ptrs, torch::Tensor flag_ptrs, int64_t counter_ptr,
                          int64_t rank, int64_t world) {
  TORCH_CHECK(world <= MAX_RANKS && world >= 2, "world must be 2..8");
  TORCH_CHECK(data_ptrs.numel() == world && flag_ptrs.numel() == world, "need one ptr per rank");
  CommView cv{};
  auto* dp = data_ptrs.data_ptr<int64_t>();
  auto* fp = flag_ptrs.data_ptr<int64_t>();
  for (int r = 0; r < world; ++r) {
    cv.data[r] = reinterpret_cast<__nv_bfloat16*>(dp[r]);
    cv.flags[r] = reinterpret_cast<uint32_t*>(fp[r]);
  }
  cv.counter = reinterpret_cast<uint32_t*>(counter_ptr);
  cv.rank = (int)rank;
  cv.world = (int)world;
  return cv;
}

#define CHECK_ROW(t, n)                                                              \
  TORCH_CHECK((t).is_cuda() && (t).scalar_type() == torch::kBFloat16 && (t).size(-1) == H, \
              n " must be CUDA bf16 [*, 6144]")

// PDL-aware launcher: plain <<<>>> when off; cudaLaunchKernelEx + programmatic stream
// serialization when on (records into CUDA graphs under stream capture — the flashinfer/
// TRT-LLM production pattern).
template <typename K, typename... Args>
static void launch_kern_t(K kern, int N, int threads, cudaStream_t stream, bool pdl, Args... args) {
  if (!pdl) {
    kern<<<N, threads, 0, stream>>>(args...);
    return;
  }
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(N);
  cfg.blockDim = dim3(threads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, args...));
}

template <typename K, typename... Args>
static void launch_kern(K kern, int N, cudaStream_t stream, bool pdl, Args... args) {
  launch_kern_t(kern, N, THREADS, stream, pdl, args...);
}

void ar_add_rmsnorm(torch::Tensor partial_in, torch::Tensor residual_in, torch::Tensor weight,
                    torch::Tensor residual_out, torch::Tensor norm_out, torch::Tensor data_ptrs,
                    torch::Tensor flag_ptrs, int64_t counter_ptr, int64_t rank, int64_t world,
                    double eps, double weight_bias, int64_t lamport, int64_t pdl) {
  CHECK_ROW(partial_in, "partial_in"); CHECK_ROW(residual_in, "residual_in");
  const int N = (int)partial_in.size(0);
  TORCH_CHECK(N <= MAX_SLOTS, "N exceeds MAX_SLOTS — take the stock path for prefill-sized batches");
  CommView cv = make_view(data_ptrs, flag_ptrs, counter_ptr, rank, world);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto go = [&](auto kern, bool p) {
    launch_kern(kern, N, stream.stream(), p,
        reinterpret_cast<__nv_bfloat16 const*>(partial_in.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(residual_in.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(norm_out.data_ptr()), cv, (float)eps, (float)weight_bias, N);
  };
  TORCH_CHECK(lamport >= 0 && lamport <= 2, "mode must be 0(seq)/1(lamport)/2(twoshot)");
  switch (cv.world * 8 + (int)lamport * 2 + (pdl ? 1 : 0)) {
    case 2 * 8 + 0 * 2 + 0: go(ar_add_rmsnorm_kernel<2, 0, false>, false); break;
    case 2 * 8 + 0 * 2 + 1: go(ar_add_rmsnorm_kernel<2, 0, true>, true); break;
    case 2 * 8 + 1 * 2 + 0: go(ar_add_rmsnorm_kernel<2, 1, false>, false); break;
    case 2 * 8 + 1 * 2 + 1: go(ar_add_rmsnorm_kernel<2, 1, true>, true); break;
    case 2 * 8 + 2 * 2 + 0: go(ar_add_rmsnorm_kernel<2, 2, false>, false); break;
    case 2 * 8 + 2 * 2 + 1: go(ar_add_rmsnorm_kernel<2, 2, true>, true); break;
    case 4 * 8 + 0 * 2 + 0: go(ar_add_rmsnorm_kernel<4, 0, false>, false); break;
    case 4 * 8 + 0 * 2 + 1: go(ar_add_rmsnorm_kernel<4, 0, true>, true); break;
    case 4 * 8 + 1 * 2 + 0: go(ar_add_rmsnorm_kernel<4, 1, false>, false); break;
    case 4 * 8 + 1 * 2 + 1: go(ar_add_rmsnorm_kernel<4, 1, true>, true); break;
    case 4 * 8 + 2 * 2 + 0: go(ar_add_rmsnorm_kernel<4, 2, false>, false); break;
    case 4 * 8 + 2 * 2 + 1: go(ar_add_rmsnorm_kernel<4, 2, true>, true); break;
    case 8 * 8 + 0 * 2 + 0: go(ar_add_rmsnorm_kernel<8, 0, false>, false); break;
    case 8 * 8 + 0 * 2 + 1: go(ar_add_rmsnorm_kernel<8, 0, true>, true); break;
    case 8 * 8 + 1 * 2 + 0: go(ar_add_rmsnorm_kernel<8, 1, false>, false); break;
    case 8 * 8 + 1 * 2 + 1: go(ar_add_rmsnorm_kernel<8, 1, true>, true); break;
    case 8 * 8 + 2 * 2 + 0: go(ar_add_rmsnorm_kernel<8, 2, false>, false); break;
    case 8 * 8 + 2 * 2 + 1: go(ar_add_rmsnorm_kernel<8, 2, true>, true); break;
    default: TORCH_CHECK(false, "unsupported world size");
  }
}

void moe_finalize_ar_add_rmsnorm(torch::Tensor gemm2_out, torch::Tensor idx, torch::Tensor scales,
                                 c10::optional<torch::Tensor> shared_out, torch::Tensor residual_in,
                                 torch::Tensor weight, torch::Tensor residual_out,
                                 torch::Tensor norm_out, torch::Tensor data_ptrs,
                                 torch::Tensor flag_ptrs, int64_t counter_ptr, int64_t rank,
                                 int64_t world, double routed_scaling, double eps,
                                 double weight_bias, int64_t lamport, int64_t pdl) {
  CHECK_ROW(gemm2_out, "gemm2_out"); CHECK_ROW(residual_in, "residual_in");
  TORCH_CHECK(idx.scalar_type() == torch::kInt32 && scales.scalar_type() == torch::kFloat32,
              "idx must be int32, scales fp32");
  const int N = (int)idx.size(0);
  const int K = (int)idx.size(1);
  TORCH_CHECK(N <= MAX_SLOTS, "N exceeds MAX_SLOTS — take the stock path for prefill-sized batches");
  TORCH_CHECK(K == 4, "M3 is top-4; other TOPK need a new instantiation");
  CommView cv = make_view(data_ptrs, flag_ptrs, counter_ptr, rank, world);
  auto stream = at::cuda::getCurrentCUDAStream();
  const __nv_bfloat16* shared_p =
      shared_out.has_value() ? reinterpret_cast<__nv_bfloat16 const*>(shared_out->data_ptr()) : nullptr;
  auto go = [&](auto kern, bool p) {
    launch_kern(kern, N, stream.stream(), p,
        reinterpret_cast<__nv_bfloat16 const*>(gemm2_out.data_ptr()), idx.data_ptr<int32_t>(),
        scales.data_ptr<float>(), shared_p,
        reinterpret_cast<__nv_bfloat16 const*>(residual_in.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(norm_out.data_ptr()), cv, (float)routed_scaling, (float)eps,
        (float)weight_bias, /*idx_kmajor=*/0, /*exp_rows=*/N, N);
  };
  TORCH_CHECK(lamport >= 0 && lamport <= 2, "mode must be 0(seq)/1(lamport)/2(twoshot)");
  switch (cv.world * 8 + (int)lamport * 2 + (pdl ? 1 : 0)) {
    case 2 * 8 + 0 * 2 + 0: go(moe_finalize_ar_add_rmsnorm_kernel<2, 4, 0, false>, false); break;
    case 2 * 8 + 0 * 2 + 1: go(moe_finalize_ar_add_rmsnorm_kernel<2, 4, 0, true>, true); break;
    case 2 * 8 + 1 * 2 + 0: go(moe_finalize_ar_add_rmsnorm_kernel<2, 4, 1, false>, false); break;
    case 2 * 8 + 1 * 2 + 1: go(moe_finalize_ar_add_rmsnorm_kernel<2, 4, 1, true>, true); break;
    case 2 * 8 + 2 * 2 + 0: go(moe_finalize_ar_add_rmsnorm_kernel<2, 4, 2, false>, false); break;
    case 2 * 8 + 2 * 2 + 1: go(moe_finalize_ar_add_rmsnorm_kernel<2, 4, 2, true>, true); break;
    case 4 * 8 + 0 * 2 + 0: go(moe_finalize_ar_add_rmsnorm_kernel<4, 4, 0, false>, false); break;
    case 4 * 8 + 0 * 2 + 1: go(moe_finalize_ar_add_rmsnorm_kernel<4, 4, 0, true>, true); break;
    case 4 * 8 + 1 * 2 + 0: go(moe_finalize_ar_add_rmsnorm_kernel<4, 4, 1, false>, false); break;
    case 4 * 8 + 1 * 2 + 1: go(moe_finalize_ar_add_rmsnorm_kernel<4, 4, 1, true>, true); break;
    case 4 * 8 + 2 * 2 + 0: go(moe_finalize_ar_add_rmsnorm_kernel<4, 4, 2, false>, false); break;
    case 4 * 8 + 2 * 2 + 1: go(moe_finalize_ar_add_rmsnorm_kernel<4, 4, 2, true>, true); break;
    case 8 * 8 + 0 * 2 + 0: go(moe_finalize_ar_add_rmsnorm_kernel<8, 4, 0, false>, false); break;
    case 8 * 8 + 0 * 2 + 1: go(moe_finalize_ar_add_rmsnorm_kernel<8, 4, 0, true>, true); break;
    case 8 * 8 + 1 * 2 + 0: go(moe_finalize_ar_add_rmsnorm_kernel<8, 4, 1, false>, false); break;
    case 8 * 8 + 1 * 2 + 1: go(moe_finalize_ar_add_rmsnorm_kernel<8, 4, 1, true>, true); break;
    case 8 * 8 + 2 * 2 + 0: go(moe_finalize_ar_add_rmsnorm_kernel<8, 4, 2, false>, false); break;
    case 8 * 8 + 2 * 2 + 1: go(moe_finalize_ar_add_rmsnorm_kernel<8, 4, 2, true>, true); break;
    default: TORCH_CHECK(false, "unsupported world size");
  }
}

// Deep-seam entry: raw device pointers exported by the fe_export flashinfer patch
// (gemm2 unfused output + TRT-LLM k-major row map + token_final_scales). The scales
// already carry routed_scaling (folded into topk_weights by sglang topk) and the
// fused shared expert rides in the expert pool (K=5 on M3), so no shared/scaling args.
void moe_epilogue_ptrs(int64_t gemm2_ptr, int64_t idx_ptr, int64_t scales_ptr, int64_t n_tokens,
                       int64_t exp_rows, int64_t topk, int64_t idx_kmajor,
                       torch::Tensor residual_in, torch::Tensor weight,
                       torch::Tensor residual_out, torch::Tensor norm_out,
                       torch::Tensor data_ptrs, torch::Tensor flag_ptrs, int64_t counter_ptr,
                       int64_t rank, int64_t world, double eps, double weight_bias,
                       int64_t lamport, int64_t pdl) {
  CHECK_ROW(residual_in, "residual_in");
  const int N = (int)n_tokens;
  const int K = (int)topk;
  TORCH_CHECK(N >= 1 && N <= MAX_SLOTS, "N out of range");
  TORCH_CHECK(exp_rows >= N, "export rows must cover the consume rows");
  TORCH_CHECK(gemm2_ptr && idx_ptr && scales_ptr, "null export pointer");
  CommView cv = make_view(data_ptrs, flag_ptrs, counter_ptr, rank, world);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto go = [&](auto kern, bool p) {
    launch_kern(kern, N, stream.stream(), p,
        reinterpret_cast<__nv_bfloat16 const*>(gemm2_ptr),
        reinterpret_cast<int32_t const*>(idx_ptr), reinterpret_cast<float const*>(scales_ptr),
        static_cast<__nv_bfloat16 const*>(nullptr),
        reinterpret_cast<__nv_bfloat16 const*>(residual_in.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(norm_out.data_ptr()), cv, /*routed_scaling=*/1.0f,
        (float)eps, (float)weight_bias, (int)idx_kmajor, (int)exp_rows, N);
  };
  TORCH_CHECK(lamport >= 0 && lamport <= 2, "mode must be 0(seq)/1(lamport)/2(twoshot)");
#define FE_MOE_K_CASES(W, KK)                                                                     \
  case W * 100 + KK * 10 + 0 * 2 + 0:                                                            \
    go(moe_finalize_ar_add_rmsnorm_kernel<W, KK, 0, false>, false); break;                       \
  case W * 100 + KK * 10 + 0 * 2 + 1:                                                            \
    go(moe_finalize_ar_add_rmsnorm_kernel<W, KK, 0, true>, true); break;                         \
  case W * 100 + KK * 10 + 1 * 2 + 0:                                                            \
    go(moe_finalize_ar_add_rmsnorm_kernel<W, KK, 1, false>, false); break;                       \
  case W * 100 + KK * 10 + 1 * 2 + 1:                                                            \
    go(moe_finalize_ar_add_rmsnorm_kernel<W, KK, 1, true>, true); break;                         \
  case W * 100 + KK * 10 + 2 * 2 + 0:                                                            \
    go(moe_finalize_ar_add_rmsnorm_kernel<W, KK, 2, false>, false); break;                       \
  case W * 100 + KK * 10 + 2 * 2 + 1:                                                            \
    go(moe_finalize_ar_add_rmsnorm_kernel<W, KK, 2, true>, true); break;
  switch ((int)world * 100 + K * 10 + (int)lamport * 2 + (pdl ? 1 : 0)) {
    FE_MOE_K_CASES(2, 4)
    FE_MOE_K_CASES(2, 5)
    FE_MOE_K_CASES(4, 4)
    FE_MOE_K_CASES(4, 5)
    FE_MOE_K_CASES(8, 4)
    FE_MOE_K_CASES(8, 5)
    default: TORCH_CHECK(false, "unsupported (world, topk) combo: ", world, ", ", K);
  }
#undef FE_MOE_K_CASES
}

// v7 window entry: one call per gemm2 column chunk, ascending chunk order on ONE stream.
// ssq_ptr = fp32 [MAX_SLOTS, MAX_NC] LOCAL slots (fused_ops allocates). Instantiated for
// TP4 (the M3 rig) x COLW {3072 (n_c=2), 1536 (n_c=4)} x TOPK {4,5} x mode {1,2} x PDL.
void moe_epilogue_ptrs_win(int64_t gemm2_ptr, int64_t g2_ld, int64_t idx_ptr, int64_t scales_ptr,
                           int64_t n_tokens, int64_t exp_rows, int64_t topk, int64_t idx_kmajor,
                           int64_t col_off, int64_t col_w, int64_t chunk_idx, int64_t n_c,
                           int64_t ssq_ptr, torch::Tensor residual_in, torch::Tensor weight,
                           torch::Tensor residual_out, torch::Tensor norm_out,
                           torch::Tensor data_ptrs, torch::Tensor flag_ptrs, int64_t counter_ptr,
                           int64_t rank, int64_t world, double eps, double weight_bias,
                           int64_t lamport, int64_t pdl) {
  CHECK_ROW(residual_in, "residual_in");
  const int N = (int)n_tokens;
  const int K = (int)topk;
  TORCH_CHECK(N >= 1 && N <= MAX_SLOTS, "N out of range");
  TORCH_CHECK(exp_rows >= N, "export rows must cover the consume rows");
  TORCH_CHECK(gemm2_ptr && idx_ptr && scales_ptr && ssq_ptr, "null pointer");
  TORCH_CHECK(lamport == 1 || lamport == 2, "window consumers are Lamport-only (mode 1|2)");
  TORCH_CHECK(n_c >= 2 && n_c <= MAX_NC, "n_c must be 2..MAX_NC");
  TORCH_CHECK(col_w * n_c == H, "chunks must tile H exactly");
  TORCH_CHECK(chunk_idx >= 0 && chunk_idx < n_c && col_off == chunk_idx * col_w, "bad window");
  TORCH_CHECK(g2_ld >= col_w, "plane row stride below window width");
  CommView cv = make_view(data_ptrs, flag_ptrs, counter_ptr, rank, world);
  TORCH_CHECK(cv.world == 4, "window path instantiated for TP4 only (add NRANKS cases)");
  const int is_last = (chunk_idx == n_c - 1) ? 1 : 0;
  const int nthreads = (int)(col_w / VEC);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto go = [&](auto kern, bool p) {
    launch_kern_t(kern, N, nthreads, stream.stream(), p,
        reinterpret_cast<__nv_bfloat16 const*>(gemm2_ptr), g2_ld,
        reinterpret_cast<int32_t const*>(idx_ptr), reinterpret_cast<float const*>(scales_ptr),
        reinterpret_cast<__nv_bfloat16 const*>(residual_in.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(norm_out.data_ptr()), cv, (float)eps,
        (float)weight_bias, (int)idx_kmajor, (int)exp_rows, (int)col_off,
        reinterpret_cast<float*>(ssq_ptr), (int)chunk_idx, (int)n_c, is_last, N);
  };
#define FE_WIN_K_CASES(CW, KK)                                                                  \
  case CW * 100 + KK * 10 + 1 * 2 + 0:                                                         \
    go(moe_finalize_ar_add_rmsnorm_win_kernel<4, KK, 1, false, CW>, false); break;             \
  case CW * 100 + KK * 10 + 1 * 2 + 1:                                                         \
    go(moe_finalize_ar_add_rmsnorm_win_kernel<4, KK, 1, true, CW>, true); break;               \
  case CW * 100 + KK * 10 + 2 * 2 + 0:                                                         \
    go(moe_finalize_ar_add_rmsnorm_win_kernel<4, KK, 2, false, CW>, false); break;             \
  case CW * 100 + KK * 10 + 2 * 2 + 1:                                                         \
    go(moe_finalize_ar_add_rmsnorm_win_kernel<4, KK, 2, true, CW>, true); break;
  switch ((int)col_w * 100 + K * 10 + (int)lamport * 2 + (pdl ? 1 : 0)) {
    FE_WIN_K_CASES(3072, 4)
    FE_WIN_K_CASES(3072, 5)
    FE_WIN_K_CASES(1536, 4)
    FE_WIN_K_CASES(1536, 5)
    default: TORCH_CHECK(false, "unsupported (col_w, topk) combo: ", col_w, ", ", K);
  }
#undef FE_WIN_K_CASES
}

// v7: make the CURRENT torch stream wait on a raw cudaEvent_t exported by the fe_chunk csrc
// patch (per-chunk gemm2 completion). Under capture this becomes the cross-stream fork edge.
void stream_wait_event(int64_t ev_handle) {
  TORCH_CHECK(ev_handle, "null event handle");
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(),
                                     reinterpret_cast<cudaEvent_t>(ev_handle), 0));
}

// IPC helpers so the Python side stays allocator-agnostic (buffers MUST be cudaMalloc'd /
// default-caching-allocator tensors; expandable_segments VMM breaks IPC — see header).
torch::Tensor get_ipc_handle(int64_t dptr) {
  cudaIpcMemHandle_t h;
  C10_CUDA_CHECK(cudaIpcGetMemHandle(&h, reinterpret_cast<void*>(dptr)));
  auto out = torch::empty({(int64_t)sizeof(h)}, torch::kUInt8);
  memcpy(out.data_ptr(), &h, sizeof(h));
  return out;
}
int64_t open_ipc_handle(torch::Tensor handle) {
  TORCH_CHECK(handle.numel() == sizeof(cudaIpcMemHandle_t), "bad handle size");
  cudaIpcMemHandle_t h;
  memcpy(&h, handle.data_ptr(), sizeof(h));
  void* p = nullptr;
  C10_CUDA_CHECK(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<int64_t>(p);
}

}  // namespace fused_epilogue

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ar_add_rmsnorm", &fused_epilogue::ar_add_rmsnorm,
        "one-shot AR + residual + RMSNorm (TP2/4/8, bf16 H=6144; lamport=0 seq flags, 1 Lamport)");
  m.def("moe_finalize_ar_add_rmsnorm", &fused_epilogue::moe_finalize_ar_add_rmsnorm,
        "MoE finalize (+shared) + one-shot AR + residual + RMSNorm (lamport=0 seq flags, 1 Lamport)");
  m.def("moe_epilogue_ptrs", &fused_epilogue::moe_epilogue_ptrs,
        "deep-seam: finalize+AR+add+RMSNorm from fe_export raw pointers (k-major trtllm row map)");
  m.def("moe_epilogue_ptrs_win", &fused_epilogue::moe_epilogue_ptrs_win,
        "v7 rung-3: consume ONE column chunk of a fe_chunk'd gemm2 export (Lamport-only; "
        "call chunks in ascending order on one stream; last chunk norms the full row)");
  m.def("stream_wait_event", &fused_epilogue::stream_wait_event,
        "cudaStreamWaitEvent(current torch stream, raw event handle) — fe_chunk fork edge");
  m.def("get_ipc_handle", &fused_epilogue::get_ipc_handle);
  m.def("open_ipc_handle", &fused_epilogue::open_ipc_handle);
  m.attr("H") = fused_epilogue::H;
  m.attr("MAX_SLOTS") = fused_epilogue::MAX_SLOTS;
  m.attr("MAX_RANKS") = fused_epilogue::MAX_RANKS;
  m.attr("NBUFS") = fused_epilogue::NBUFS;
  m.attr("MAX_NC") = fused_epilogue::MAX_NC;
}
