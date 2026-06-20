"""Arenas — the per-model mapping of {sglang version, image, seam subset, KL floors}.

The validator's job is to score a kernel inside a SPECIFIC model's sglang. Different
models need different runtimes (gpt-oss/Qwen on the pinned build, DeepSeek-V4 on a
Blackwell sglang, a launch-window model on a nightly). An *arena* captures everything
that is model-specific so that "try a new model" is a CONFIG ROW here, not a manual
sglang checkout + a hand-edited constant + a worklog note.

``docker_image`` is an **Optima-OWNED validator image**, not a vendor sglang container.
Consensus requires every validator to measure on a BYTE-IDENTICAL runtime, and a stock
``lmsysorg/sglang`` image has neither the Optima seam (the ``.pth`` bootstrap + the
package) nor the calibrated gates — two validators on the "same" vendor image would
still score differently. So the arena pins the *whole* scoring stack — sglang +
CUDA/PyTorch + the Optima seam + the model kwargs + the calibrated floors — built into
one image (e.g. ``ghcr.io/optima/validator:<arena>``). A miner may develop inside a
vendor container (we enforce nothing on miners); the *validator* runs the Optima image.

Two facts this reconciles (they look opposed but aren't):

* **Consensus wants ONE runtime.** Validators measuring the same kernel on different
  sglang versions diverge under Yuma consensus, so only ONE arena is *competed* at a
  time (one pinned runtime fleet-wide per season). The arena registry does NOT mean
  "run 5 pins concurrently in production" — that would N× the divergence surface.
* **Dev/rotation wants MANY.** Trying M3 today must not endanger the validated
  gpt-oss path. Declaring an arena lets you stage the next rotation target (its image,
  version, calibrated floors) without disturbing the live one.

So: arenas are the declarative source of "what runtime + calibration does this model
use"; the ACTIVE arena is the one being scored. ``DEFAULT_ARENA`` equals today's
behavior, so this module is additive — nothing changes until you pass ``--arena``.

Arenas live strictly BELOW the SlotSpec waist: the four invariants and the miner
contract never move. The ``seam_adapters`` field is a NAME subset of the one seam
table (``optima/seams.py``), never a parallel list. ``kl_floors`` is the per-MODEL
calibrated KL gate that overrides the model-agnostic ``SlotSpec.kl_threshold`` default
(README calibration findings: the floor is model-specific — gpt-oss 3.9e-4 vs
attention ~6e-3 vs det/non-det ~30×, so one per-slot constant can't be right for two
models at once).

stdlib-only (like ``seams.py``): importable from the ``.pth`` bootstrap and ``compat``
without pulling in torch.

Adding a model = one row. Example (fill in from the model's image):

    MINIMAX_M3 = Arena(
        name="minimax-m3",
        model_path="MiniMaxAI/MiniMax-M3",        # the HF id you serve
        sglang_version="0.5.13",                  # the sglang baked into the image below
        docker_image="ghcr.io/optima/validator:minimax-m3",  # Optima-owned (sglang+seam+gates)
        seam_adapters=("attention", "moe", "collective"),  # the seams that apply to M3
        kl_floors={"attention.decode": 0.04},     # calibrate on the first clean run
        engine_kwargs={"tp_size": 4, "moe_runner_backend": "triton"},
        notes="Launch-window target; calibrate floors on the pod.",
    )
    # then add it to ARENAS below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Arena:
    name: str  # short id, used as --arena <name> and stamped on scores
    model_path: str  # the model served/scored in this arena ("" = generic/any)
    sglang_version: str  # the pinned sglang for THIS model (the per-arena consensus version)
    # Optima-OWNED validator image pinning the WHOLE scoring runtime (sglang + CUDA/torch +
    # the Optima seam + the calibrated gates), not a vendor sglang container — consensus
    # needs a byte-identical runtime across validators. "" = the host venv (default arena).
    docker_image: str = ""
    # NAME subset of seams.SEAM_ADAPTERS that apply to this model; () = all of them.
    seam_adapters: tuple[str, ...] = ()
    # slot name -> calibrated mean-KL floor for THIS model (overrides SlotSpec.kl_threshold).
    kl_floors: dict = field(default_factory=dict)
    # base sglang.Engine kwargs this model needs (tp_size, moe_runner_backend, …); explicit
    # CLI flags still override these.
    engine_kwargs: dict = field(default_factory=dict)
    notes: str = ""

    def kl_floor_for(self, slot_name: str) -> Optional[float]:
        """The per-model calibrated KL floor for a slot, or None to fall back to the
        slot's default / the CLI value."""
        return self.kl_floors.get(slot_name)

    def applies_seam(self, adapter_name: str) -> bool:
        """Whether a seam adapter (by name) is in scope for this arena. Empty subset = all."""
        return (not self.seam_adapters) or (adapter_name in self.seam_adapters)

    def competable(self) -> bool:
        """An arena can be SCORED only once its pinned runtime is filled in. A declared
        stub (empty sglang_version) can exist for planning but must not gate/score."""
        return bool(self.sglang_version)


# The validated path on the pinned sglang — equals pre-arena behavior. PINNED_SGLANG
# (optima/compat.py) aliases this version, so all existing code keeps working unchanged.
DEFAULT_ARENA = Arena(
    name="default",
    model_path="",  # generic: the gpt-oss / Qwen2.5 dev path validated on the pin
    sglang_version="0.5.12.post1",
    docker_image="",
    seam_adapters=(),  # all five
    kl_floors={},  # fall back to the per-slot SlotSpec defaults
    engine_kwargs={},
    notes="Validated gpt-oss/Qwen path on the pinned sglang; equals pre-arena behavior.",
)

# The registry. Add a model = add a row (see the module docstring template).
ARENAS: dict[str, Arena] = {DEFAULT_ARENA.name: DEFAULT_ARENA}


def get_arena(name: Optional[str]) -> Arena:
    """Resolve an arena by name; ``None``/"" -> the default arena."""
    if not name:
        return DEFAULT_ARENA
    try:
        return ARENAS[name]
    except KeyError:
        known = ", ".join(sorted(ARENAS)) or "(none)"
        raise KeyError(f"unknown arena {name!r}; known arenas: {known}") from None


def arena_for_model(model_path: str) -> Arena:
    """The arena whose ``model_path`` matches, else the default (generic) arena."""
    for a in ARENAS.values():
        if a.model_path and a.model_path == model_path:
            return a
    return DEFAULT_ARENA


def list_arenas() -> list[str]:
    return sorted(ARENAS)
