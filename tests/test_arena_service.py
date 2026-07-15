from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from optima.arena_service import (
    SCREEN_STAGES,
    AdmissionDecision,
    ArenaCandidateBinding,
    ArenaCapacityPolicy,
    ArenaQualificationWork,
    ArenaQueueSnapshot,
    ArenaRuntimeIdentity,
    ArenaService,
    ArenaServiceError,
    ArenaServiceManifest,
    ArenaServiceRegistry,
    NonCrownScreenPolicy,
    PromotionDecision,
    ScreenGrade,
    ScreenStagePolicy,
    ScreenStageResult,
    ServingShape,
    WorkloadMixture,
    WorkloadRegime,
)
from optima.bundle_hash import content_hash
from optima.chain.publication import publish_worker_bundle
from optima.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationPlanFactory,
    QualificationReservation,
)


def _h(label: str) -> str:
    return (label.encode().hex() + "1" * 64)[:64]


def _manifest(**changes) -> ArenaServiceManifest:
    runtime = ArenaRuntimeIdentity(
        arena_id="minimax-m3-sm120",
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("engine"),
        validator_overlay_digest=_h("overlay"),
        worker_distribution_digest=_h("worker"),
        model_revision_digest=_h("revision"),
        model_manifest_digest=_h("manifest"),
        model_content_digest=_h("model"),
        target_architecture="sm120",
        topology_class="pcie-switch",
        topology_digest=_h("topology"),
        gpu_count=8,
        tensor_parallel_size=8,
    )
    workload = WorkloadMixture(
        _h("corpus"),
        "finalized-entropy-v1",
        (
            WorkloadRegime(
                "decode-serving",
                "decode",
                600_000,
                (ServingShape(256, 256, 8, 32), ServingShape(1024, 128, 4, 16)),
            ),
            WorkloadRegime(
                "long-prefill-serving",
                "long_prefill",
                400_000,
                (ServingShape(8192, 32, 1, 16), ServingShape(32768, 16, 1, 8)),
            ),
        ),
    )
    capacity = ArenaCapacityPolicy(64, 600, 4, 8, 4, 2, 4, 3)
    screens = NonCrownScreenPolicy(
        tuple(ScreenStagePolicy(stage, 1_000) for stage in SCREEN_STAGES)
    )
    values = {
        "runtime": runtime,
        "workload": workload,
        "capacity": capacity,
        "screens": screens,
        "qualification_policy_digest": _h("qualification-policy"),
        "provider_digest": _h("provider"),
    }
    values.update(changes)
    return ArenaServiceManifest(**values)


def _publication(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir(parents=True)
    (source / "manifest.toml").write_text("bundle_id = 'candidate'\n")
    source.chmod(0o700)
    (source / "manifest.toml").chmod(0o600)
    committed = content_hash(source)
    return publish_worker_bundle(source, tmp_path / "publications", committed)


def _binding(tmp_path: Path, *, attempt: int = 1) -> ArenaCandidateBinding:
    publication = _publication(tmp_path)
    reservation = QualificationReservation(
        _h("reservation"),
        publication.digest,
        "attention.msa-prefill",
        _h("delta"),
        0,
        "miner-hotkey",
        10,
        0,
        0,
        ("attention.msa-prefill",),
    )
    return ArenaCandidateBinding(reservation, publication, attempt)


class _Provider:
    provider_digest = _h("provider")

    def __init__(self, grades=None, *, wrong_stage: bool = False):
        self.grades = dict(grades or {})
        self.wrong_stage = wrong_stage
        self.screen_calls = []
        self.plan_calls = []

    def run_screen(self, manifest, stage, candidate):
        self.screen_calls.append((manifest.digest, stage.stage, candidate.digest))
        name = "build" if self.wrong_stage and stage.stage == "static" else stage.stage
        grade, elapsed = self.grades.get(stage.stage, (ScreenGrade.PASS, 10))
        return ScreenStageResult(name, grade, _h(stage.stage), elapsed)

    def build_qualification(self, request, state=None):
        self.plan_calls.append(request)
        reservations = tuple(row.reservation for row in request.candidates)
        authority = QualificationAuthorityManifest(
            "registered",
            _h("authority"),
            _h("source"),
            _h("commitment"),
            _h("secret"),
            tuple(row.selected_delta_digest for row in reservations),
            reservations,
        )
        factory = QualificationPlanFactory(
            authority, lambda _ref: b"s" * 32, lambda _s: None
        )
        return ArenaQualificationWork(
            factory,
            object(),
            lambda *_: None,
            lambda **_: None,
            99.0,
            request.qualification_policy_digest,
        )


def test_service_identity_binds_every_serving_authority() -> None:
    manifest = _manifest()
    assert len(manifest.digest) == 64
    assert manifest.service_id == f"minimax-m3-sm120@{manifest.digest}"
    assert manifest.screens.to_dict()["crownable"] is False

    variants = (
        dataclasses.replace(
            manifest,
            runtime=dataclasses.replace(manifest.runtime, runtime_digest=_h("runtime2")),
        ),
        dataclasses.replace(
            manifest,
            runtime=dataclasses.replace(
                manifest.runtime, model_content_digest=_h("model2")
            ),
        ),
        dataclasses.replace(
            manifest,
            runtime=dataclasses.replace(
                manifest.runtime, topology_digest=_h("topology2")
            ),
        ),
        dataclasses.replace(
            manifest,
            workload=dataclasses.replace(manifest.workload, prompt_corpus_digest=_h("new")),
        ),
        dataclasses.replace(
            manifest,
            capacity=dataclasses.replace(manifest.capacity, max_queue_depth=65),
        ),
        dataclasses.replace(
            manifest, qualification_policy_digest=_h("qualification-policy2")
        ),
        dataclasses.replace(manifest, provider_digest=_h("provider2")),
    )
    assert len({manifest.digest, *(row.digest for row in variants)}) == 8


def test_workload_requires_exact_decode_and_long_prefill_mixture() -> None:
    decode = WorkloadRegime(
        "decode", "decode", 1_000_000, (ServingShape(1, 1, 1, 1),)
    )
    with pytest.raises(ArenaServiceError, match="decode and long_prefill"):
        WorkloadMixture(_h("corpus"), "seed-v1", (decode,))

    prefill = WorkloadRegime(
        "prefill", "long_prefill", 1, (ServingShape(8192, 1, 1, 1),)
    )
    with pytest.raises(ArenaServiceError, match="1M ppm"):
        WorkloadMixture(_h("corpus"), "seed-v1", (decode, prefill))


def test_admission_is_capacity_bounded_and_fail_closed() -> None:
    service = ArenaService(_manifest(), _Provider())
    assert service.admit(ArenaQueueSnapshot(0, 0, 0, 0)) is AdmissionDecision.ADMIT
    assert service.admit(ArenaQueueSnapshot(1, 1, 4, 0)) is AdmissionDecision.QUEUE
    assert service.admit(ArenaQueueSnapshot(64, 1, 0, 0)) is AdmissionDecision.HOLD
    assert service.admit(ArenaQueueSnapshot(1, 600, 0, 0)) is AdmissionDecision.HOLD
    assert (
        service.admit_qualification(
            ArenaQueueSnapshot(0, 0, 0, 5), cohort_size=4
        )
        is AdmissionDecision.QUEUE
    )
    assert (
        service.admit_qualification(
            ArenaQueueSnapshot(0, 0, 0, 0), cohort_size=5
        )
        is AdmissionDecision.HOLD
    )
    assert (
        service.retry_disposition("qualification", attempt=3)
        is PromotionDecision.RETRY
    )
    assert (
        service.retry_disposition("qualification", attempt=4)
        is PromotionDecision.HOLD
    )
    assert (
        service.retry_disposition("infrastructure", attempt=3)
        is PromotionDecision.HOLD
    )


def test_registry_is_closed_and_unambiguous() -> None:
    service = ArenaService(_manifest(), _Provider())
    registry = ArenaServiceRegistry((service,))
    assert registry.require("minimax-m3-sm120") is service
    assert len(registry.digest) == 64
    with pytest.raises(ArenaServiceError, match="not registered"):
        registry.require("miner-chosen-arena")
    with pytest.raises(ArenaServiceError, match="empty or ambiguous"):
        ArenaServiceRegistry((service, service))


def test_provider_is_validator_supplied_and_digest_bound() -> None:
    provider = _Provider()
    provider.provider_digest = _h("forged")
    with pytest.raises(ArenaServiceError, match="implementation identity"):
        ArenaService(_manifest(), provider)


def test_all_non_crown_screens_promote_in_fixed_order(tmp_path: Path) -> None:
    provider = _Provider()
    service = ArenaService(_manifest(), provider)
    binding = _binding(tmp_path)
    receipt = service.screen(binding)

    assert receipt.decision is PromotionDecision.PROMOTE
    assert tuple(row.stage for row in receipt.results) == SCREEN_STAGES
    assert tuple(row[1] for row in provider.screen_calls) == SCREEN_STAGES
    assert set(receipt.to_dict()) == {
        "candidate_digest",
        "decision",
        "results",
        "screen_attempt",
        "service_digest",
    }
    assert all(
        forbidden not in repr(receipt.to_dict()).lower()
        for forbidden in ("speedup", "score", "crown")
    )


def test_fail_rejects_and_no_decision_retries_then_holds(tmp_path: Path) -> None:
    binding = _binding(tmp_path / "fail")
    provider = _Provider({"abi": (ScreenGrade.FAIL, 10)})
    receipt = ArenaService(_manifest(), provider).screen(binding)
    assert receipt.decision is PromotionDecision.REJECT
    assert tuple(row.stage for row in receipt.results) == ("static", "build", "abi")

    retry_binding = _binding(tmp_path / "retry", attempt=1)
    timeout = _Provider({"build": (ScreenGrade.PASS, 1_001)})
    retry = ArenaService(_manifest(), timeout).screen(retry_binding)
    assert retry.results[-1].grade is ScreenGrade.NO_DECISION
    assert retry.decision is PromotionDecision.RETRY

    hold_binding = _binding(tmp_path / "hold", attempt=2)
    hold = ArenaService(_manifest(), timeout).screen(hold_binding)
    assert hold.decision is PromotionDecision.HOLD


def test_provider_cannot_substitute_a_screen_stage(tmp_path: Path) -> None:
    service = ArenaService(_manifest(), _Provider(wrong_stage=True))
    with pytest.raises(ArenaServiceError, match="changed the requested screen stage"):
        service.screen(_binding(tmp_path))


def test_only_exact_promoted_coverage_reaches_qualification(tmp_path: Path) -> None:
    provider = _Provider()
    service = ArenaService(_manifest(), provider)
    binding = _binding(tmp_path)
    promoted = service.screen(binding)
    work = service.plan_qualification((binding,), (promoted,))
    assert type(work) is ArenaQualificationWork
    assert work.factory.manifest.reservations == (binding.reservation,)
    assert provider.plan_calls[0].service_digest == service.identity

    rejected_provider = _Provider({"static": (ScreenGrade.FAIL, 1)})
    rejected_service = ArenaService(_manifest(), rejected_provider)
    rejected = rejected_service.screen(binding)
    with pytest.raises(ArenaServiceError, match="promoted coverage"):
        rejected_service.plan_qualification((binding,), (rejected,))

    other = dataclasses.replace(promoted, service_digest=_h("other-service"))
    with pytest.raises(ArenaServiceError, match="promoted coverage"):
        service.plan_qualification((binding,), (other,))


def test_provider_cannot_change_finalized_order(tmp_path: Path) -> None:
    class WrongProvider(_Provider):
        def build_qualification(self, request, state=None):
            work = super().build_qualification(request, state)
            wrong = dataclasses.replace(
                request.candidates[0].reservation, arrival_order=9
            )
            authority = dataclasses.replace(work.factory.manifest, reservations=(wrong,))
            return dataclasses.replace(
                work,
                factory=QualificationPlanFactory(
                    authority, work.factory.secret_loader, work.factory.plan_builder
                ),
            )

    provider = WrongProvider()
    service = ArenaService(_manifest(), provider)
    binding = _binding(tmp_path)
    receipt = service.screen(binding)
    with pytest.raises(ArenaServiceError, match="finalized qualification order"):
        service.plan_qualification((binding,), (receipt,))


def test_provider_cannot_change_registered_qualification_policy(tmp_path: Path) -> None:
    class WrongPolicyProvider(_Provider):
        def build_qualification(self, request, state=None):
            return dataclasses.replace(
                super().build_qualification(request, state),
                qualification_policy_digest=_h("other-qualification-policy"),
            )

    service = ArenaService(_manifest(), WrongPolicyProvider())
    binding = _binding(tmp_path)
    receipt = service.screen(binding)
    with pytest.raises(ArenaServiceError, match="qualification policy"):
        service.plan_qualification((binding,), (receipt,))


def test_arena_service_has_no_dynamic_import_authority() -> None:
    source = Path(__file__).parents[1].joinpath("optima", "arena_service.py").read_text()
    assert "importlib" not in source
    assert "import_module" not in source
    assert "entry_point" not in source
