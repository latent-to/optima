"""Near-copy fingerprint + cumulative cross-round copy detection.

Pins the two confirmed gaps: (1) a reformatted/recommented copy that flips the
exact content hash is still caught, and (2) copy detection now spans rounds, so a
copy revealed in a LATER round than the original is no longer mislabeled original.
"""

from pathlib import Path

from optima.commit_reveal import Ledger, make_commitment
from optima.copy_fingerprint import (
    bundle_fingerprint,
    bundle_slot_file_fingerprints,
    bundle_slot_fingerprints,
    normalized_source,
    source_fingerprint,
    structural_fingerprint,
)

# A rename-everything + constant-tweak copy of ORIG: same structure, vars renamed
# (x->inp, out->dst, d->half), the // 2 constant changed to // 3. No statements added.
RENAMED_TWEAKED = '''\
import torch

def silu_and_mul(inp, dst):
    half = inp.shape[-1] // 3
    dst.copy_(torch.nn.functional.silu(inp[..., :half]) * inp[..., half:])
'''

ORIG = '''\
"""A kernel docstring."""
import torch

def silu_and_mul(x, out):
    # compute the gate
    d = x.shape[-1] // 2
    out.copy_(torch.nn.functional.silu(x[..., :d]) * x[..., d:])
'''

# Same logic, reflowed: different whitespace, different comments, no docstring,
# extra blank lines and parens. A byte hash differs; the structure does not.
REFORMATTED = '''\
import torch


def silu_and_mul(x, out):
    d = (x.shape[-1] // 2)
    # totally different comment wording here
    out.copy_((torch.nn.functional.silu(x[..., :d]) * x[..., d:]))
'''

# A genuine logic change (drops the silu) -> must NOT collide with ORIG.
DIFFERENT = '''\
import torch

def silu_and_mul(x, out):
    d = x.shape[-1] // 2
    out.copy_(x[..., :d] * x[..., d:])
'''


def test_reformat_recomment_redocstring_fingerprints_identical():
    assert source_fingerprint(ORIG) == source_fingerprint(REFORMATTED)


def test_genuine_logic_change_fingerprints_differently():
    assert source_fingerprint(ORIG) != source_fingerprint(DIFFERENT)


def test_normalized_source_strips_docstring_and_comments():
    n = normalized_source(ORIG)
    assert "A kernel docstring" not in n
    assert "compute the gate" not in n  # comments gone


def test_bundle_fingerprint_on_a_real_example_is_stable_nonempty():
    bundle = Path(__file__).resolve().parent.parent / "examples" / "miner_silu_triton"
    fp = bundle_fingerprint(bundle)
    assert fp and len(fp) == 64
    assert bundle_fingerprint(bundle) == fp  # deterministic


def _commit_reveal(led: Ledger, hotkey: str, ch: str, salt: str, rnd: int, fp: str):
    led.commit(hotkey, make_commitment(ch, hotkey, salt), rnd)
    return led.reveal(hotkey, ch, salt, rnd, fingerprint=fp)


def test_near_copy_in_a_later_round_is_flagged():
    led = Ledger()
    F = "fingerprint-A"
    a = _commit_reveal(led, "alice", "HASH_ORIG", "s", 0, F)
    # bob reflows alice's kernel: NEW exact hash, SAME fingerprint, LATER round.
    b = _commit_reveal(led, "bob", "HASH_REFLOW", "s", 1, F)
    assert a.original is True
    assert b.original is False  # caught as a near-copy across rounds


def test_exact_copy_in_a_later_round_is_flagged():
    led = Ledger()
    a = _commit_reveal(led, "alice", "HASH_X", "s", 0, "fp1")
    c = _commit_reveal(led, "carol", "HASH_X", "s", 2, "fp1")
    assert a.original is True
    assert c.original is False  # cross-round exact copy now caught (was a gap)


def test_same_hotkey_resubmitting_own_work_is_not_a_copy():
    led = Ledger()
    a0 = _commit_reveal(led, "alice", "HASH_X", "s0", 0, "fpA")
    a1 = _commit_reveal(led, "alice", "HASH_X", "s1", 1, "fpA")
    assert a0.original is True
    assert a1.original is True  # you can't plagiarize yourself


def test_independent_distinct_kernels_both_original():
    led = Ledger()
    a = _commit_reveal(led, "alice", "HASH_A", "s", 0, "fpA")
    b = _commit_reveal(led, "bob", "HASH_B", "s", 0, "fpB")
    assert a.original and b.original


# ---- structural (advisory) fingerprint: catches rename + constant-tweak ----


def test_structural_fingerprint_survives_rename_and_constant_tweak():
    # The normalized fingerprint differs (names/constants changed), but the structural
    # skeleton is identical -> advisory near-copy signal the normalized form misses.
    assert source_fingerprint(ORIG) != source_fingerprint(RENAMED_TWEAKED)
    assert structural_fingerprint(ORIG) == structural_fingerprint(RENAMED_TWEAKED)


def test_structural_fingerprint_distinguishes_different_ops():
    # silu vs a plain multiply (DIFFERENT) must NOT collide structurally.
    assert structural_fingerprint(ORIG) != structural_fingerprint(DIFFERENT)


def test_structural_advisory_is_not_auto_demote():
    led = Ledger()
    led.commit("alice", make_commitment("H_A", "alice", "s"), 0)
    led.reveal("alice", "H_A", "s", 0, fingerprint="fpA", structural_fingerprint="SKEL")
    # bob: different exact + normalized fp, but SAME structural skeleton.
    matches = led.structural_near_copies("SKEL", "bob")
    led.commit("bob", make_commitment("H_B", "bob", "s"), 1)
    bob = led.reveal("bob", "H_B", "s", 1, fingerprint="fpB", structural_fingerprint="SKEL")
    assert matches == ["alice"]      # surfaced for review
    assert bob.original is True      # but NOT demoted (advisory only)


# ---- relocation / padding evasion (per-slot + per-file compares) ----


def _write_bundle(root: Path, ops: list[tuple[str, str, str]], files: dict[str, str]) -> Path:
    """Materialize a minimal bundle: ``ops`` = (slot, source, entry) rows; ``files`` =
    relpath -> python source."""
    root.mkdir(parents=True, exist_ok=True)
    lines = ['bundle_id = "t"', 'abi_version = "optima-op-abi-v0"', ""]
    for slot, source, entry in ops:
        lines += ["[[ops]]", f'slot = "{slot}"', f'source = "{source}"', f'entry = "{entry}"', ""]
    (root / "manifest.toml").write_text("\n".join(lines))
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
    return root


def _reveal_bundle(led: Ledger, hotkey: str, ch: str, rnd: int, bundle: Path):
    led.commit(hotkey, make_commitment(ch, hotkey, "s"), rnd)
    return led.reveal(
        hotkey, ch, "s", rnd,
        fingerprint=bundle_fingerprint(bundle),
        slot_fingerprints=bundle_slot_fingerprints(bundle),
        slot_file_fingerprints=bundle_slot_file_fingerprints(bundle),
    )


def test_relocated_body_behind_a_reexport_is_still_flagged(tmp_path):
    # alice: the body lives in the DECLARED entry module.
    a = _write_bundle(tmp_path / "a", [("activation.silu_and_mul", "kernels/silu.py", "silu_and_mul")],
                      {"kernels/silu.py": ORIG})
    # bob: entry is a one-line re-export; alice's body (reflowed) hides in an imported
    # module at a DIFFERENT path — the exact-hash, whole-bundle AND per-slot closure
    # fingerprints all differ, so only the per-file containment compare can catch it.
    b = _write_bundle(tmp_path / "b", [("activation.silu_and_mul", "kernels/silu.py", "silu_and_mul")],
                      {"kernels/silu.py": "from ._impl import silu_and_mul\n",
                       "kernels/_impl.py": REFORMATTED})
    led = Ledger()
    ra = _reveal_bundle(led, "alice", "H_A", 0, a)
    rb = _reveal_bundle(led, "bob", "H_B", 1, b)
    assert ra.original is True
    assert rb.original is False


def test_padding_an_extra_op_does_not_evade_demotion(tmp_path):
    a = _write_bundle(tmp_path / "a", [("activation.silu_and_mul", "kernels/silu.py", "silu_and_mul")],
                      {"kernels/silu.py": ORIG})
    # bob pads the stolen (reflowed, relocated) slot with a second unrelated op so the
    # whole-bundle fingerprint can never match alice's single-op bundle.
    pad = "import torch\n\ndef rmsnorm(x, w, out, eps):\n    v = (x * x).mean(-1, keepdim=True)\n    out.copy_(x * torch.rsqrt(v + eps) * w)\n"
    b = _write_bundle(tmp_path / "b",
                      [("activation.silu_and_mul", "kernels/main.py", "silu_and_mul"),
                       ("norm.rmsnorm", "kernels/rms.py", "rmsnorm")],
                      {"kernels/main.py": REFORMATTED, "kernels/rms.py": pad})
    led = Ledger()
    ra = _reveal_bundle(led, "alice", "H_A", 0, a)
    rb = _reveal_bundle(led, "bob", "H_B", 1, b)
    assert ra.original is True
    assert rb.original is False


def test_shared_vendored_utility_alone_is_not_a_copy(tmp_path):
    # Both miners vendor the SAME public helper next to genuinely different kernels:
    # file-set INTERSECTION is non-empty but neither set CONTAINS the other -> no demote.
    util = ("import torch\n\ndef ceil_div(a, b):\n    return (a + b - 1) // b\n\n"
            "def pad_to(x, m):\n    r = x.shape[-1] % m\n    return x if r == 0 else "
            "torch.nn.functional.pad(x, (0, m - r))\n")
    a = _write_bundle(tmp_path / "a", [("activation.silu_and_mul", "kernels/k.py", "silu_and_mul")],
                      {"kernels/k.py": "from .util import ceil_div\n" + ORIG, "kernels/util.py": util})
    b = _write_bundle(tmp_path / "b", [("activation.silu_and_mul", "kernels/k.py", "silu_and_mul")],
                      {"kernels/k.py": "from .util import ceil_div\n" + DIFFERENT, "kernels/util.py": util})
    led = Ledger()
    ra = _reveal_bundle(led, "alice", "H_A", 0, a)
    rb = _reveal_bundle(led, "bob", "H_B", 1, b)
    assert ra.original is True
    assert rb.original is True


def test_file_fingerprints_skip_boilerplate_and_follow_imports(tmp_path):
    b = _write_bundle(tmp_path / "b", [("activation.silu_and_mul", "kernels/silu.py", "silu_and_mul")],
                      {"kernels/silu.py": "from ._impl import silu_and_mul\n",
                       "kernels/_impl.py": ORIG})
    file_fps = bundle_slot_file_fingerprints(b)["activation.silu_and_mul"]
    # the one-line re-export shim is boilerplate (below the substantial floor); the
    # imported body is followed and fingerprinted path-independently.
    assert file_fps == [source_fingerprint(ORIG)]
