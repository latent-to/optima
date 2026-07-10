"""Per-epoch prompt sampling.

A kernel must not be able to special-case a fixed handful of prompts, so the
validator samples a fresh subset each epoch from a larger corpus, keyed by an
epoch seed. In production this corpus would be drawn from the real (agentic)
serving distribution and rotated/expanded each epoch; this is a stand-in that is
diverse enough to exercise varied shapes and stabilize the KL estimate.
"""

from __future__ import annotations

import hashlib
import random

PROMPT_ENGINE_VERSION: int = 1
"""Bump when the corpus or sampling changes. Folded into the block seed so a version
change reshuffles prompts even at the same block — old and new prompt sets never
collide, and the engine version a score was produced under stays reproducible."""

CORPUS: tuple[str, ...] = (
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain, step by step, how a hash map handles collisions.",
    "Summarize the tradeoffs between TCP and UDP for a real-time game.",
    "Given a list of integers, describe an O(n) algorithm to find the majority element.",
    "Refactor a nested callback chain into async/await and explain why.",
    "What are the failure modes of two-phase commit, and how does Paxos help?",
    "Implement binary search and state its preconditions and invariants.",
    "Describe how a B-tree keeps itself balanced on insertion.",
    "Explain the CAP theorem with a concrete example for each pair.",
    "Walk through how TLS establishes a session key.",
    "Compare mutexes and channels for sharing state between threads.",
    "Explain how a generational garbage collector decides what to collect.",
    "Given a directed graph, outline Tarjan's algorithm for strongly connected components.",
    "Describe the memory hierarchy and why cache-oblivious algorithms matter.",
    "Explain backpropagation through a single linear layer with a bias.",
    "What is the difference between bagging and boosting, with an example each?",
    "Outline a rate limiter using a token bucket and discuss burst handling.",
    "Explain MVCC and how it avoids read locks in a database.",
    "Describe how consistent hashing reduces churn when a node leaves.",
    "Write a SQL query to find the second-highest salary per department.",
    "Explain how a bloom filter trades memory for false positives.",
    "Describe the actor model and where it fits versus shared memory.",
    "How does a CPU branch predictor work, and what is a misprediction penalty?",
    "Explain the difference between latency and throughput with an analogy.",
    "Outline how Raft elects a leader and commits a log entry.",
    "Describe how copy-on-write makes fork cheap.",
    "Explain what makes a hash function suitable for a hash table vs cryptography.",
    "Give an example where eventual consistency is acceptable and one where it is not.",
    "Describe how a JIT compiler decides what to optimize at runtime.",
    "Explain vectorization and when the compiler can and cannot do it for you.",
    "Walk through quicksort and explain the worst case and how to avoid it.",
    "Explain how attention computes a weighted sum and why it scales as O(n^2).",
    "Describe how paging and a TLB translate a virtual address.",
    "What is the difference between optimistic and pessimistic concurrency control?",
    "Explain how a reverse proxy and a load balancer differ in purpose.",
    "Describe the tradeoffs of column-oriented vs row-oriented storage.",
    "Explain how gradient checkpointing trades compute for memory.",
    "Outline how a merge sort can be parallelized across cores.",
    "Explain what a race condition is and give a minimal example.",
    "Describe how speculative decoding speeds up autoregressive generation.",
)


_APPROX_TOKENS_PER_SENTENCE = 18  # corpus sentences + the per-instance salt, roughly


def _long_prompt(rng: random.Random, input_len: int) -> str:
    """One synthetic long prompt of ~``input_len`` tokens (approximate by design —
    the scorer is a PAIRED A/B, so what matters is that both arms see the identical
    workload, not that the count is tokenizer-exact).

    Two properties are load-bearing (learned from the real-transcript 256k prompt set):
      * PREFIX-DISJOINT: a per-prompt salt header means no two prompts share a prefix,
        so concurrent-request throughput can't be inflated by radix-cache hits.
      * NO REPEATED BLOCKS: every sentence instance carries its own salt, so the KV
        cache never contains exact duplicate blocks a kernel could special-case (and
        long-context block scoring sees realistic, non-degenerate keys).
    """
    parts = [f"[case {rng.getrandbits(64):016x}] Read the following notes, then answer the final question."]
    approx = 2 * _APPROX_TOKENS_PER_SENTENCE
    while approx < input_len:
        parts.append(f"Note {rng.getrandbits(32):08x}: {rng.choice(CORPUS)}")
        approx += _APPROX_TOKENS_PER_SENTENCE
    parts.append("Question: summarize the three most important ideas from the notes above.")
    return " ".join(parts)


def sample_prompts(n: int, seed: int, input_len: int | None = None) -> list[str]:
    """Deterministically sample ``n`` prompts for an epoch.

    Without replacement when ``n <= len(CORPUS)``, otherwise with replacement so
    callers can request large workloads for throughput measurement.

    ``input_len`` (approximate tokens) switches to the LONG-PROMPT engine: without it
    the corpus averages 10-20 tokens per prompt, so the measured regime is pure decode
    and a prefill-side win (e.g. the MSA prefill indexer, ~30% of long-context serving
    prefill) is INVISIBLE to the scorer. Long-context serving is prefill-dominated
    (~91% of wall in the 2026-07-10 M3 256k workload), so arenas that sell that regime
    must score it. Same determinism/rotation contract as the short corpus; production
    intent remains a hosted real-distribution corpus (this is the stand-in).
    """
    rng = random.Random(seed)
    if input_len is not None and input_len > 0:
        return [_long_prompt(rng, input_len) for _ in range(n)]
    if n <= len(CORPUS):
        return rng.sample(list(CORPUS), n)
    return [rng.choice(CORPUS) for _ in range(n)]


def derive_seed(block_hash: str, *, version: int = PROMPT_ENGINE_VERSION) -> int:
    """Map a chain block hash to a deterministic 64-bit epoch seed.

    Two validators that score a submission at the same block derive the *same* seed,
    so they draw the identical prompt set — a hard requirement for cross-validator
    consensus (they must agree on weights for that submission). The seed also rotates
    unpredictably per block, so a kernel cannot pre-bake answers for a known prompt
    set. ``version`` is folded in so a prompt-engine bump reshuffles even at the same
    block.
    """
    digest = hashlib.sha256(f"v{version}:{block_hash}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def sample_prompts_for_block(block_hash: str, n: int, *,
                             version: int = PROMPT_ENGINE_VERSION) -> list[str]:
    """Block-hash-seeded prompts: identical across validators at a given block,
    unpredictable across blocks. Thin wrapper over ``sample_prompts``."""
    return sample_prompts(n, derive_seed(block_hash, version=version))
