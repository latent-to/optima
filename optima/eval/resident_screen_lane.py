"""The resident hot-swap lane behind the ``abbreviated_serving`` screen stage.

This is the bridge between two call shapes that cannot meet directly:

* the intake controller screens candidates ONE AT A TIME as they arrive
  (``ArenaService.screen`` -> ``ArenaServiceProvider.run_screen``), while
* :meth:`~optima.eval.oci_backend.OCIEngineExecutor.execute_resident` owns the
  call stack for a WHOLE engine lifetime (one driver callback, one engine
  load, any number of candidates).

:class:`ResidentScreenLane` resolves that: it runs each lifetime on a
background thread whose driver serves a work queue, so the engine stays
loaded between arrivals — the inference-service shape — and each
``screen`` call simply hands its candidate to whichever lifetime is live,
lazily booting one when none is.  Lifetimes recycle on the screen loop's own
stop conditions (candidate budget, canary drift); a budget recycle re-screens
the candidate on the fresh lifetime automatically, while a canary withdrawal
is surfaced to the caller because evidence was consumed.

Trust tier: screen/routing only, exactly like the queue module underneath.
Non-swappable bundles (AOT device artifacts, dep-patched trees) never enter
the lane: :class:`ResidentServingScreenStage` records an explicit waiver and
routes them to qualification, which is the deciding authority either way.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Callable, Protocol, Sequence

from optima.arena_service import (
    ArenaCandidateBinding,
    ScreenGrade,
    ScreenStageResult,
)
from optima.eval.oci_backend import stage_swap_bundle
from optima.eval.resident_queue import (
    CandidateScreenVerdict,
    ResidentScreenLoop,
    ScreenCandidate,
    ScreenPolicy,
    ScreenSession,
)
from optima.manifest import Manifest, load_manifest
from optima.stack_identity import canonical_digest


SERVING_SCREEN_STAGE = "abbreviated_serving"
_STAGE_EVIDENCE_SCHEMA = "optima.arena.resident-screen-stage.v1"
_WAIVER_EVIDENCE_SCHEMA = "optima.arena.resident-screen-waiver.v1"


class ResidentScreenLaneError(RuntimeError):
    """The lane, its lifetime, or a stage input is invalid or failed."""


def screen_swappability(manifest: Manifest) -> str | None:
    """Why this bundle cannot hot-swap into a live engine, or None.

    Mirrors the worker-side refusals in :func:`optima.seam.swap_resident_bundle`
    so routing decisions are made before any engine is touched.
    """

    if type(manifest) is not Manifest:
        raise ResidentScreenLaneError("swappability requires a typed manifest")
    if any(op.aot_exports for op in manifest.ops):
        return "aot device artifacts are not swappable in the screen tier"
    if manifest.dep_patches:
        return "dep-patched bundles are not swappable in the screen tier"
    return None


class ResidentLifetimeFactory(Protocol):
    """Opens one resident engine lifetime and blocks until it closes.

    Deployment code closes this over the backend and its launch authority —
    typically a thin wrapper around ``executor.execute_resident`` (see
    :func:`make_backend_lifetime_factory`).  The callable must invoke
    ``driver`` exactly once with a started session and return the lifetime's
    execution evidence.
    """

    def __call__(self, driver: Callable[[ScreenSession], object]) -> object: ...


def make_backend_lifetime_factory(
    executor: object,
    launch: object,
    binding: object,
    mount: object,
    plan: object,
    *,
    swap_intake_root: str | Path,
    deadline_provider: Callable[[], float],
) -> ResidentLifetimeFactory:
    """Bind one backend's ``execute_resident`` into a lane lifetime factory."""

    execute = getattr(executor, "execute_resident", None)
    if not callable(execute) or not callable(deadline_provider):
        raise ResidentScreenLaneError("lifetime factory authorities are not callable")
    root = Path(swap_intake_root)

    def factory(driver: Callable[[ScreenSession], object]):
        return execute(
            launch,
            binding,
            mount,
            plan,
            deadline=deadline_provider(),
            swap_intake_root=root,
            driver=driver,
        )

    return factory


class _Work:
    __slots__ = ("candidate", "done", "verdict", "error")

    def __init__(self, candidate: ScreenCandidate) -> None:
        self.candidate = candidate
        self.done = threading.Event()
        self.verdict: CandidateScreenVerdict | None = None
        self.error: BaseException | None = None


_CLOSE = object()


class ResidentScreenLane:
    """Serializes screen candidates onto long-lived resident engine lifetimes.

    One caller at a time: ``screen``/``close`` hold the lane lock for their
    whole duration (the intake controller is sequential by design), so there
    is never a queue backlog — each lifetime serves at most one in-flight
    candidate.  The engine stays loaded while the lane idles between
    arrivals; only the screen loop's own stop conditions or ``close`` end a
    lifetime.
    """

    def __init__(
        self,
        lifetime_factory: ResidentLifetimeFactory,
        *,
        prompts: Sequence[str],
        policy: ScreenPolicy = ScreenPolicy(),
        verdict_timeout_s: float = 3600.0,
        close_timeout_s: float = 1800.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(lifetime_factory) or not callable(clock):
            raise ResidentScreenLaneError("lane authorities are not callable")
        if type(policy) is not ScreenPolicy:
            raise ResidentScreenLaneError("lane screen policy has the wrong type")
        prompt_plan = tuple(prompts)
        if not prompt_plan or any(
            not isinstance(row, str) or not row for row in prompt_plan
        ):
            raise ResidentScreenLaneError("lane prompt plan is empty or untyped")
        for name, value in (
            ("verdict_timeout_s", verdict_timeout_s),
            ("close_timeout_s", close_timeout_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 < float(value) <= 86_400
            ):
                raise ResidentScreenLaneError(f"lane {name} is invalid")
        self._factory = lifetime_factory
        self._prompts = prompt_plan
        self._policy = policy
        self._verdict_timeout_s = float(verdict_timeout_s)
        self._close_timeout_s = float(close_timeout_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._state = threading.Lock()
        self._thread: threading.Thread | None = None
        self._work: queue.Queue = queue.Queue()
        self._session_id: str | None = None
        self._lifetime_error: BaseException | None = None
        self._last_evidence: object | None = None
        self._closed = False

    @property
    def session_id(self) -> str | None:
        """The most recent lifetime's session identity (survives its close)."""
        with self._state:
            return self._session_id

    @property
    def last_lifetime_evidence(self) -> object | None:
        with self._state:
            return self._last_evidence

    def screen(self, candidate: ScreenCandidate) -> CandidateScreenVerdict:
        """Screen one candidate on the live lifetime, booting one if needed.

        A budget-exhausted lifetime recycles transparently (the candidate was
        untouched); a canary-withdrawn verdict is returned as-is because its
        brackets were spent — the caller's retry machinery re-screens it.
        """

        if type(candidate) is not ScreenCandidate:
            raise ResidentScreenLaneError("lane candidate is not exactly typed")
        with self._lock:
            if self._closed:
                raise ResidentScreenLaneError("resident screen lane is closed")
            for _attempt in range(2):
                self._ensure_lifetime()
                item = _Work(candidate)
                self._work.put(item)
                self._await(item)
                if item.error is not None:
                    self._join_lifetime()
                    raise ResidentScreenLaneError(
                        f"resident screen lifetime failed: {item.error}"
                    ) from item.error
                if item.verdict is not None:
                    if item.verdict.withdrawn:
                        # The driver broke out of its loop; reap the thread so
                        # the next arrival opens a fresh lifetime.
                        self._join_lifetime()
                    return item.verdict
                # None: the lifetime's candidate budget was exhausted before
                # this candidate ran.  Recycle and retry once on fresh state.
                self._join_lifetime()
            raise ResidentScreenLaneError(
                "a fresh resident lifetime refused the candidate"
            )

    def close(self) -> None:
        """Permanently close the lane, ending any live lifetime."""
        with self._lock:
            self._closed = True
            thread = self._thread
            if thread is not None and thread.is_alive():
                self._work.put(_CLOSE)
                thread.join(self._close_timeout_s)
                if thread.is_alive():
                    raise ResidentScreenLaneError(
                        "resident lifetime did not close in time"
                    )
            self._thread = None

    def _ensure_lifetime(self) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            return
        with self._state:
            self._lifetime_error = None
        # A fresh queue per lifetime: nothing stale can leak into the new
        # driver, and a sentinel left in a dead queue is simply dropped.
        self._work = queue.Queue()
        self._thread = threading.Thread(
            target=self._run_lifetime,
            args=(self._work,),
            name="optima-resident-screen",
            daemon=True,
        )
        self._thread.start()

    def _run_lifetime(self, work: queue.Queue) -> None:
        def driver(session):
            loop = ResidentScreenLoop(
                session, prompts=self._prompts, policy=self._policy
            )
            with self._state:
                self._session_id = session.session_id
            while True:
                item = work.get()
                if item is _CLOSE:
                    break
                try:
                    item.verdict = loop.screen(item.candidate)
                except BaseException as exc:
                    item.error = exc
                    item.done.set()
                    raise
                item.done.set()
                if item.verdict is None or loop.stopped_reason is not None:
                    break
            # Lifetimes open lazily on the first candidate, and every screened
            # candidate begins with the opening stock read, so a finishing
            # session always has at least one batch behind it.
            return session.finish()

        try:
            # The factory is reviewed deployment code around execute_resident,
            # which already enforces its own evidence type on return.
            evidence = self._factory(driver)
            with self._state:
                self._last_evidence = evidence
        except BaseException as exc:  # surfaced to the waiter via _await
            with self._state:
                self._lifetime_error = exc

    def _await(self, item: _Work) -> None:
        deadline = self._clock() + self._verdict_timeout_s
        while not item.done.wait(timeout=0.25):
            thread = self._thread
            if thread is None or not thread.is_alive():
                with self._state:
                    error = self._lifetime_error
                raise ResidentScreenLaneError(
                    f"resident lifetime died before the verdict: {error}"
                ) from error
            if self._clock() > deadline:
                raise ResidentScreenLaneError("resident screen verdict timed out")

    def _join_lifetime(self) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(self._close_timeout_s)
            if thread.is_alive():
                raise ResidentScreenLaneError(
                    "resident lifetime did not close in time"
                )
        self._thread = None


class ResidentServingScreenStage:
    """The ``abbreviated_serving`` stage engine for an arena provider.

    Swappable bundles are staged into the content-addressed swap intake and
    screened through the resident lane; the verdict maps to the stage grade
    (confident pass -> PASS, confident regression or wrong dispatch -> FAIL,
    noise or withdrawn evidence -> NO_DECISION, retried by the controller's
    screen budgets).  Non-swappable bundles receive an explicitly recorded
    waiver PASS: the screen tier cannot pre-price them cheaply, so
    qualification — the deciding authority for every candidate — prices them
    on its own dedicated launches.
    """

    stage = SERVING_SCREEN_STAGE

    def __init__(
        self,
        lane: ResidentScreenLane,
        swap_intake_root: str | Path,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(lane) is not ResidentScreenLane or not callable(clock):
            raise ResidentScreenLaneError("stage authorities are not exact")
        self._lane = lane
        self._root = Path(swap_intake_root)
        self._clock = clock

    def run_screen(self, candidate: ArenaCandidateBinding) -> ScreenStageResult:
        if type(candidate) is not ArenaCandidateBinding:
            raise ResidentScreenLaneError("stage candidate is not exactly typed")
        started = self._clock()
        publication = candidate.publication
        manifest = load_manifest(publication.root)
        refusal = screen_swappability(manifest)
        if refusal is not None:
            evidence = canonical_digest(
                _WAIVER_EVIDENCE_SCHEMA,
                {
                    "candidate_digest": candidate.digest,
                    "publication_content_hash": publication.content_hash,
                    "reason": refusal,
                },
            )
            return ScreenStageResult(
                self.stage, ScreenGrade.PASS, evidence, self._elapsed_ms(started)
            )
        staged_digest = stage_swap_bundle(
            self._root, publication.root, expected_digest=publication.content_hash
        )
        slots = tuple(sorted({op.slot for op in manifest.ops}))
        verdict = self._lane.screen(
            ScreenCandidate(
                candidate.reservation.reservation_digest, staged_digest, slots
            )
        )
        evidence = canonical_digest(
            _STAGE_EVIDENCE_SCHEMA,
            {
                "candidate_digest": candidate.digest,
                "publication_content_hash": publication.content_hash,
                "session_id": self._lane.session_id,
                "staged_digest": staged_digest,
                "verdict": verdict.to_dict(),
            },
        )
        return ScreenStageResult(
            self.stage,
            _stage_grade(verdict),
            evidence,
            self._elapsed_ms(started),
        )

    def _elapsed_ms(self, started: float) -> int:
        return max(1, round((self._clock() - started) * 1000))


def _stage_grade(verdict: CandidateScreenVerdict) -> ScreenGrade:
    if verdict.withdrawn:
        return ScreenGrade.NO_DECISION
    if verdict.rejected_dispatch:
        return ScreenGrade.FAIL
    if verdict.verdict is None or not verdict.verdict.confident:
        return ScreenGrade.NO_DECISION
    return ScreenGrade.PASS if verdict.passed else ScreenGrade.FAIL


__all__ = [
    "ResidentLifetimeFactory",
    "ResidentScreenLane",
    "ResidentScreenLaneError",
    "ResidentServingScreenStage",
    "SERVING_SCREEN_STAGE",
    "make_backend_lifetime_factory",
    "screen_swappability",
]
