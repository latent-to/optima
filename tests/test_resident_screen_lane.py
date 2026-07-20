"""CPU tests for the resident screen lane and the abbreviated_serving stage."""

from __future__ import annotations

import dataclasses
import threading

import pytest

from optima.arena_service import ArenaCandidateBinding, ScreenGrade
from optima.bundle_hash import content_hash
from optima.chain.publication import publish_worker_bundle
from optima.eval.oci_resident_session import ResidentBatchEvidence, SwapReceipt
from optima.eval.oci_session_protocol import BatchEvidence, PromptEvidence
from optima.eval.qualification_intake import QualificationReservation
from optima.eval.resident_queue import ScreenCandidate, ScreenPolicy
from optima.eval.resident_screen_lane import (
    ResidentScreenLane,
    ResidentScreenLaneError,
    ResidentServingScreenStage,
    screen_swappability,
)
from optima.manifest import load_manifest

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
SLOT = "activation.silu_and_mul"


def _evidence(tokens: int = 4) -> BatchEvidence:
    return BatchEvidence(
        (
            PromptEvidence(
                tuple(range(tokens)),
                tuple(((-0.5, 0),) for _ in range(tokens)),
            ),
        )
    )


class FakeResidentSession:
    """Same playback model as the queue tests, plus the session surface."""

    def __init__(
        self,
        stock_rate: float,
        candidate_rates: dict[str, float],
        slots: dict[str, tuple[str, ...]] | None = None,
        stock_drift_after: int | None = None,
        fail_on_swap: bool = False,
    ) -> None:
        self.session_id = "5" * 32
        self.stock_rate = stock_rate
        self.candidate_rates = candidate_rates
        self.slots = slots or {digest: (SLOT,) for digest in candidate_rates}
        self.stock_drift_after = stock_drift_after
        self.fail_on_swap = fail_on_swap
        self.generation = 0
        self.active: str | None = None
        self.batch_count = 0
        self.stock_reads = 0
        self.swaps: list[str | None] = []
        self.clock = 0.0
        self.finished = False

    def swap(self, bundle_digest: str | None) -> SwapReceipt:
        if self.fail_on_swap:
            raise RuntimeError("engine lost")
        self.generation += 1
        self.active = bundle_digest
        self.swaps.append(bundle_digest)
        self.clock += 30.0
        return SwapReceipt(
            len(self.swaps) - 1,
            self.generation,
            bundle_digest,
            () if bundle_digest is None else self.slots[bundle_digest],
            self.clock - 30.0,
            self.clock,
        )

    def execute_batch(self, prompts, *, canary: bool = False):
        assert not canary or self.active is None
        tokens = 1000
        if self.active is None:
            rate = self.stock_rate
            self.stock_reads += 1
            if (
                self.stock_drift_after is not None
                and self.stock_reads > self.stock_drift_after
            ):
                rate *= 0.90
        else:
            rate = self.candidate_rates[self.active]
        elapsed = tokens / rate
        started = self.clock
        self.clock += elapsed
        row = ResidentBatchEvidence(
            self.batch_count,
            f"{self.batch_count + 5:032x}",
            f"{self.batch_count + 6:032x}".replace("0", "9", 1),
            self.generation,
            () if self.active is None else self.slots[self.active],
            canary,
            started,
            self.clock,
            tokens,
            _evidence(),
        )
        self.batch_count += 1
        return row

    def finish(self) -> object:
        self.finished = True
        return ("session-evidence", self.session_id, self.batch_count)


class FakeLifetimeFactory:
    """Boots a fresh fake session per lifetime on the lane's own thread."""

    def __init__(self, builder) -> None:
        self.builder = builder
        self.calls = 0
        self.sessions: list[FakeResidentSession] = []

    def __call__(self, driver):
        self.calls += 1
        session = self.builder(self.calls)
        self.sessions.append(session)
        result = driver(session)
        return ("lifetime-evidence", session.session_id, result)


def _candidate(digest: str = DIGEST_A, name: str = "cand") -> ScreenCandidate:
    return ScreenCandidate(name, digest, (SLOT,))


def _lane(builder, **overrides) -> ResidentScreenLane:
    options = {"prompts": ("p",), "verdict_timeout_s": 30.0, "close_timeout_s": 30.0}
    options.update(overrides)
    return ResidentScreenLane(FakeLifetimeFactory(builder), **options)


class TestResidentScreenLane:
    def test_engine_stays_loaded_between_arrivals(self) -> None:
        factory = FakeLifetimeFactory(
            lambda _n: FakeResidentSession(100.0, {DIGEST_A: 112.0, DIGEST_B: 80.0})
        )
        lane = ResidentScreenLane(
            factory, prompts=("p",), verdict_timeout_s=30.0, close_timeout_s=30.0
        )
        first = lane.screen(_candidate(DIGEST_A, "a"))
        assert first.passed
        second = lane.screen(_candidate(DIGEST_B, "b"))
        assert not second.passed and second.failure is None
        # ONE engine lifetime served both arrivals — the residency property.
        assert factory.calls == 1
        assert lane.session_id == "5" * 32
        lane.close()
        assert factory.sessions[0].finished
        assert lane.last_lifetime_evidence is not None

    def test_budget_recycles_transparently(self) -> None:
        factory = FakeLifetimeFactory(
            lambda _n: FakeResidentSession(100.0, {DIGEST_A: 112.0, DIGEST_B: 112.0})
        )
        lane = ResidentScreenLane(
            factory,
            prompts=("p",),
            policy=ScreenPolicy(max_candidates_per_lifetime=1),
            verdict_timeout_s=30.0,
            close_timeout_s=30.0,
        )
        assert lane.screen(_candidate(DIGEST_A, "a")).passed
        assert lane.screen(_candidate(DIGEST_B, "b")).passed
        assert factory.calls == 2
        assert factory.sessions[0].finished
        lane.close()

    def test_withdrawn_verdict_surfaces_and_recycles(self) -> None:
        factory = FakeLifetimeFactory(
            lambda n: FakeResidentSession(
                100.0,
                {DIGEST_A: 112.0, DIGEST_B: 112.0},
                stock_drift_after=1 if n == 1 else None,
            )
        )
        lane = ResidentScreenLane(
            factory, prompts=("p",), verdict_timeout_s=30.0, close_timeout_s=30.0
        )
        first = lane.screen(_candidate(DIGEST_A, "a"))
        assert first.withdrawn and first.verdict is None
        second = lane.screen(_candidate(DIGEST_A, "a"))
        assert second.passed
        assert factory.calls == 2
        lane.close()

    def test_boot_failure_raises_then_recovers(self) -> None:
        def builder(call: int) -> FakeResidentSession:
            if call == 1:
                raise RuntimeError("no GPUs free")
            return FakeResidentSession(100.0, {DIGEST_A: 112.0})

        lane = _lane(builder)
        with pytest.raises(ResidentScreenLaneError, match="died before the verdict"):
            lane.screen(_candidate())
        assert lane.screen(_candidate()).passed
        lane.close()

    def test_screen_error_kills_lifetime_then_recovers(self) -> None:
        def builder(call: int) -> FakeResidentSession:
            return FakeResidentSession(
                100.0, {DIGEST_A: 112.0}, fail_on_swap=call == 1
            )

        lane = _lane(builder)
        with pytest.raises(ResidentScreenLaneError, match="lifetime failed"):
            lane.screen(_candidate())
        assert lane.screen(_candidate()).passed
        lane.close()

    def test_closed_lane_rejects_candidates(self) -> None:
        lane = _lane(lambda _n: FakeResidentSession(100.0, {DIGEST_A: 112.0}))
        lane.close()
        with pytest.raises(ResidentScreenLaneError, match="closed"):
            lane.screen(_candidate())

    def test_close_is_safe_before_any_lifetime(self) -> None:
        lane = _lane(lambda _n: FakeResidentSession(100.0, {DIGEST_A: 112.0}))
        lane.close()

    def test_untyped_candidate_rejected(self) -> None:
        lane = _lane(lambda _n: FakeResidentSession(100.0, {DIGEST_A: 112.0}))
        with pytest.raises(ResidentScreenLaneError, match="exactly typed"):
            lane.screen(object())  # type: ignore[arg-type]
        lane.close()


def _bundle_tree(tmp_path, *, dep_patch: bool = False):
    source = tmp_path / "source"
    kernels = source / "kernels"
    kernels.mkdir(parents=True)
    (kernels / "k.py").write_text("def k(x, out):\n    return None\n")
    lines = [
        "bundle_id = 'screen-test-bundle'",
        "abi_version = 'optima-op-abi-v0'",
    ]
    if dep_patch:
        patches = source / "patches"
        patches.mkdir()
        (patches / "x.patch").write_text(
            "--- a/f.cu\n+++ b/f.cu\n@@ -1,1 +1,1 @@\n-old\n+new\n"
        )
        lines += [
            "[[dep_patches]]",
            "target = 'flashinfer'",
            "path = 'patches/x.patch'",
        ]
    lines += [
        "[[ops]]",
        f"slot = '{SLOT}'",
        "source = 'kernels/k.py'",
        "entry = 'k'",
        "dtypes = ['bfloat16']",
    ]
    (source / "manifest.toml").write_text("\n".join(lines) + "\n")
    for path in sorted(source.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)
    source.chmod(0o700)
    return source


def _binding(tmp_path, *, dep_patch: bool = False) -> ArenaCandidateBinding:
    source = _bundle_tree(tmp_path, dep_patch=dep_patch)
    committed = content_hash(source)
    publication = publish_worker_bundle(source, tmp_path / "publications", committed)
    reservation = QualificationReservation(
        "c" * 64,
        publication.digest,
        SLOT,
        "d" * 64,
        0,
        "miner-hotkey",
        10,
        0,
        0,
        (SLOT,),
    )
    return ArenaCandidateBinding(reservation, publication, 1)


class TestScreenSwappability:
    def test_pure_source_bundle_is_swappable(self, tmp_path) -> None:
        manifest = load_manifest(_bundle_tree(tmp_path))
        assert screen_swappability(manifest) is None

    def test_dep_patched_bundle_is_not_swappable(self, tmp_path) -> None:
        manifest = load_manifest(_bundle_tree(tmp_path, dep_patch=True))
        assert "dep-patched" in screen_swappability(manifest)

    def test_aot_bundle_is_not_swappable(self, tmp_path) -> None:
        manifest = load_manifest(_bundle_tree(tmp_path))
        patched = dataclasses.replace(
            manifest,
            ops=(dataclasses.replace(manifest.ops[0], aot_exports=("aot",)),),
        )
        assert "aot" in screen_swappability(patched)


class TestResidentServingScreenStage:
    def _stage(self, tmp_path, builder):
        lane = ResidentScreenLane(
            FakeLifetimeFactory(builder),
            prompts=("p",),
            verdict_timeout_s=30.0,
            close_timeout_s=30.0,
        )
        root = tmp_path / "swap-intake"
        root.mkdir()
        return ResidentServingScreenStage(lane, root), lane, root

    def test_swappable_winner_passes(self, tmp_path) -> None:
        binding = _binding(tmp_path)
        staged = binding.publication.content_hash
        stage, lane, root = self._stage(
            tmp_path, lambda _n: FakeResidentSession(100.0, {staged: 112.0})
        )
        result = stage.run_screen(binding)
        assert result.stage == "abbreviated_serving"
        assert result.grade is ScreenGrade.PASS
        assert (root / staged).is_dir()
        lane.close()

    def test_swappable_loser_fails(self, tmp_path) -> None:
        binding = _binding(tmp_path)
        staged = binding.publication.content_hash
        stage, lane, _root = self._stage(
            tmp_path, lambda _n: FakeResidentSession(100.0, {staged: 80.0})
        )
        assert stage.run_screen(binding).grade is ScreenGrade.FAIL
        lane.close()

    def test_wrong_dispatch_fails(self, tmp_path) -> None:
        binding = _binding(tmp_path)
        staged = binding.publication.content_hash
        stage, lane, _root = self._stage(
            tmp_path,
            lambda _n: FakeResidentSession(
                100.0, {staged: 112.0}, slots={staged: ("other.slot",)}
            ),
        )
        assert stage.run_screen(binding).grade is ScreenGrade.FAIL
        lane.close()

    def test_noise_or_withdrawal_is_no_decision(self, tmp_path) -> None:
        binding = _binding(tmp_path)
        staged = binding.publication.content_hash
        stage, lane, _root = self._stage(
            tmp_path,
            lambda _n: FakeResidentSession(
                100.0, {staged: 112.0}, stock_drift_after=1
            ),
        )
        assert stage.run_screen(binding).grade is ScreenGrade.NO_DECISION
        lane.close()

    def test_unswappable_bundle_gets_waiver_pass(self, tmp_path) -> None:
        binding = _binding(tmp_path, dep_patch=True)
        factory = FakeLifetimeFactory(
            lambda _n: FakeResidentSession(100.0, {DIGEST_A: 112.0})
        )
        lane = ResidentScreenLane(
            factory, prompts=("p",), verdict_timeout_s=30.0, close_timeout_s=30.0
        )
        root = tmp_path / "swap-intake"
        root.mkdir()
        stage = ResidentServingScreenStage(lane, root)
        result = stage.run_screen(binding)
        assert result.grade is ScreenGrade.PASS
        # The lane (and therefore any engine) was never touched, and nothing
        # was staged into the swap intake.
        assert factory.calls == 0
        assert list(root.iterdir()) == []
        lane.close()
