from __future__ import annotations

import os
from pathlib import Path

import pytest

import optima.chain.validator_loop as loop
from optima.arena_service import (
    SCREEN_STAGES, AdmissionDecision, ArenaQualificationWork,
    ArenaScreenReceipt, ArenaService, ArenaServiceRegistry, PromotionDecision,
    ScreenGrade, ScreenStageResult,
)
from optima.bundle_hash import content_hash
from optima.chain import FinalizedRevealSnapshot, RevealedCommitment
from optima.chain.intake import FinalizedIntakeStore, IntakeScope
from optima.chain.payload import encode_payload
from optima.eval.evidence_store import EvidenceArtifactRef
from optima.eval.qualification import QualificationDecision
from optima.eval.qualification_intake import (
    QualificationAuthorityManifest, QualificationIntakeBatch,
    QualificationIntakeOutcome, QualificationPlanFactory,
)


BLOCK = 90
BLOCK_HASH = "0x" + "9" * 64
SCOPE = IntakeScope("0x" + "0" * 64, 307)


def _bundle(root: Path, body: str) -> Path:
    (root / "kernels").mkdir(parents=True)
    (root / "manifest.toml").write_text(
        'bundle_id = "test"\n'
        'abi_version = "optima-op-abi-v0"\n\n'
        '[[ops]]\n'
        'slot = "activation.silu_and_mul"\n'
        'source = "kernels/k.py"\n'
        'entry = "silu_and_mul"\n'
        'dtypes = ["float32"]\n'
    )
    (root / "kernels/k.py").write_text(body)
    for directory in (root, root / "kernels"):
        directory.chmod(0o700)
    for file in (root / "manifest.toml", root / "kernels/k.py"):
        file.chmod(0o600)
    return root


def _snapshot(rows: list[tuple[str, str]]) -> FinalizedRevealSnapshot:
    reveals = tuple(
        RevealedCommitment(hotkey, payload, BLOCK, BLOCK_HASH, index)
        for index, (hotkey, payload) in enumerate(rows)
    )
    return FinalizedRevealSnapshot(BLOCK, BLOCK_HASH, reveals)


class _NoWeightsSubtensor:
    def get_block_hash(self, block):
        assert block == 0
        return SCOPE.genesis_hash


def _run(tmp_path, monkeypatch, snapshot, sources, **changes):
    monkeypatch.setattr(
        loop.chain,
        "read_finalized_reveal_history",
        lambda *_, **__: snapshot,
    )
    calls = []

    def fetcher(_url, expected, _root):
        calls.append(expected)
        return sources[expected]

    monkeypatch.setattr(loop, "fetch_bundle", fetcher)

    options = dict(
        intake_db=tmp_path / "state" / "intake.sqlite3",
        private_root=tmp_path / "private-cache",
        publication_root=tmp_path / "worker",
        intake_only=True,
    )
    options.update(changes)
    return loop.run_pass(_NoWeightsSubtensor(), 307, **options), calls, options


def test_finalized_reveal_publishes_once_and_restart_reopens(tmp_path, monkeypatch):
    source = _bundle(
        tmp_path / "source",
        "def silu_and_mul(x, out):\n    out.copy_(x)\n",
    )
    digest = content_hash(source)
    snapshot = _snapshot([("miner", encode_payload(digest, "https://example.com/a"))])
    result, calls, options = _run(tmp_path, monkeypatch, snapshot, {digest: source})

    assert result.seen == 1 and len(result.reserved) == 1
    assert len(result.published) == 1 and result.decisions == {}
    assert calls == [digest]
    with FinalizedIntakeStore(options["intake_db"], scope=SCOPE) as store:
        row = store.all()[0]
        assert row.status == "published"
        assert row.publication_digest == next(iter(result.published.values()))
        assert row.arrival.content_hash == digest

    second, second_calls, _ = _run(
        tmp_path, monkeypatch, snapshot, {digest: source}
    )
    assert second.reserved == [] and second.published == {}
    assert second_calls == []


def test_malformed_finalized_payload_is_reserved_and_never_fetched(tmp_path, monkeypatch):
    snapshot = _snapshot([("miner", "not-json")])
    result, calls, options = _run(tmp_path, monkeypatch, snapshot, {})
    assert calls == [] and len(result.reserved) == 1
    assert len(result.rejected) == 1
    with FinalizedIntakeStore(options["intake_db"], scope=SCOPE) as store:
        row = store.all()[0]
        assert row.status == "failed" and row.reason == "invalid_payload"


def test_deterministically_unpublishable_submission_is_not_retried(
    tmp_path, monkeypatch
):
    source = _bundle(
        tmp_path / "source",
        "def silu_and_mul(x, out):\n    out.copy_(x)\n",
    )
    reserved = source / ".optima-native-artifact.json"
    reserved.write_text("{}\n")
    reserved.chmod(0o600)
    digest = content_hash(source)
    snapshot = _snapshot(
        [("miner", encode_payload(digest, "https://example.com/a"))]
    )
    result, _calls, options = _run(
        tmp_path, monkeypatch, snapshot, {digest: source}
    )
    assert len(result.rejected) == 1
    with FinalizedIntakeStore(options["intake_db"], scope=SCOPE) as store:
        row = store.all()[0]
        assert row.status == "failed"
        assert row.reason.startswith("publication_source:")


def test_reformatted_later_delta_is_copy_without_any_weight_edge(tmp_path, monkeypatch):
    first = _bundle(
        tmp_path / "first",
        "import torch\n\ndef silu_and_mul(x, out):\n"
        "    d = x.shape[-1] // 2\n"
        "    out.copy_(torch.nn.functional.silu(x[..., :d]) * x[..., d:])\n",
    )
    second = _bundle(
        tmp_path / "second",
        "import torch\n\n# formatting only\ndef silu_and_mul(x, out):\n"
        "    d = (x.shape[-1] // 2)\n"
        "    out.copy_((torch.nn.functional.silu(x[..., :d]) * x[..., d:]))\n",
    )
    first_hash, second_hash = content_hash(first), content_hash(second)
    assert first_hash != second_hash
    snapshot = _snapshot([
        ("author", encode_payload(first_hash, "https://example.com/a")),
        ("copycat", encode_payload(second_hash, "https://example.com/b")),
    ])
    result, _calls, options = _run(
        tmp_path,
        monkeypatch,
        snapshot,
        {first_hash: first, second_hash: second},
    )
    assert len(result.published) == 1 and len(result.copies) == 1
    with FinalizedIntakeStore(options["intake_db"], scope=SCOPE) as store:
        rows = store.all()
        assert [row.status for row in rows] == ["published", "failed"]
        assert rows[1].reason.startswith("copy_of:")


def test_live_loop_calls_batch_qualification_and_retains_fail_outcome(
    tmp_path, monkeypatch
):
    source = _bundle(
        tmp_path / "source",
        "def silu_and_mul(x, out):\n    out.copy_(x)\n",
    )
    digest = content_hash(source)
    snapshot = _snapshot([("miner", encode_payload(digest, "https://example.com/a"))])

    calls = []
    service = object.__new__(ArenaService)
    service.manifest = type(
        "Manifest",
        (),
        {"digest": "e" * 64, "qualification_policy_digest": "f" * 64},
    )()
    registry = object.__new__(ArenaServiceRegistry)
    monkeypatch.setattr(ArenaServiceRegistry, "require", lambda *_: service)
    monkeypatch.setattr(ArenaService, "admit", lambda *_: AdmissionDecision.ADMIT)
    monkeypatch.setattr(
        ArenaService, "admit_qualification", lambda *_args, **_kwargs: AdmissionDecision.ADMIT
    )
    monkeypatch.setattr(
        ArenaService,
        "screen",
        lambda self, candidate: ArenaScreenReceipt(
            self.identity,
            candidate.digest,
            candidate.screen_attempt,
            tuple(
                ScreenStageResult(stage, ScreenGrade.PASS, chr(97 + index) * 64, 1)
                for index, stage in enumerate(SCREEN_STAGES)
            ),
            PromotionDecision.PROMOTE,
        ),
    )

    def plan(_self, candidates, _receipts, state=None):
        reservations = tuple(row.reservation for row in candidates)
        authority = QualificationAuthorityManifest(
            "registered", "a" * 64, "b" * 64, "c" * 64, "d" * 64,
            tuple(row.selected_delta_digest for row in reservations), reservations,
        )
        factory = QualificationPlanFactory(
            authority, lambda _ref: b"s" * 32, lambda _secret: None
        )
        return ArenaQualificationWork(
            factory,
            object(),
            lambda *_: None,
            lambda **_: None,
            30.0,
            _self.manifest.qualification_policy_digest,
        )

    monkeypatch.setattr(ArenaService, "plan_qualification", plan)
    # The focused test uses a deliberately non-building plan and a mocked runner.
    monkeypatch.setattr(loop, "QualificationAuthorityManifest", type("NotManifest", (), {}))

    def qualify(factory, **_kwargs):
        calls.append(factory.manifest.digest)
        authority = factory.manifest.reservations[0]
        outcome = QualificationIntakeOutcome(
            authority.reservation_digest,
            authority.selected_delta_digest,
            factory.manifest.digest,
            QualificationDecision.FAIL,
            "rejected",
            False,
            attempt_artifact_sha256="b" * 64,
            report_digest="c" * 64,
        )
        ref = EvidenceArtifactRef(
            "qualification.cohort-attempt", "b" * 64, 1,
            "application/json", "optima.qualification.cohort-attempt.v1",
        )
        return QualificationIntakeBatch(factory.manifest.digest, (outcome,), ref)

    monkeypatch.setattr(loop, "run_qualification_intake", qualify)
    result, _fetches, options = _run(
        tmp_path,
        monkeypatch,
        snapshot,
        {digest: source},
        intake_only=False,
        arena_registry=registry,
        arena_id="test-arena",
    )
    assert len(calls) == 1 and len(calls[0]) == 64
    assert set(result.decisions.values()) == {"FAIL"}
    with FinalizedIntakeStore(options["intake_db"], scope=SCOPE) as store:
        row = store.all()[0]
        assert row.status == "failed" and row.decision == "FAIL"
        assert store.qualification_dispositions(row.reservation_id)[0]["decision"] == "FAIL"


def test_once_mode_propagates_validator_fault(monkeypatch, tmp_path):
    monkeypatch.setattr(
        loop,
        "run_pass",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("finality failed")),
    )
    with pytest.raises(RuntimeError, match="finality failed"):
        loop.run_validator(
            _NoWeightsSubtensor(),
            307,
            intake_db=tmp_path / "state.sqlite3",
            private_root=tmp_path / "private",
            publication_root=tmp_path / "worker",
            once=True,
        )


def test_settlement_refreshes_stale_pass_height_before_leasing():
    class Store:
        lease_blocks = []

        def has_pending_settlement(self):
            return True

        def lease_settlement_cohort(self, *, current_block):
            self.lease_blocks.append(current_block)
            return None

    store = Store()
    assert loop._settle_pending(
        store,
        current_block=BLOCK,
        finalized_block_provider=lambda: BLOCK + 100,
    ) == {}
    assert store.lease_blocks == [BLOCK + 100]


def test_settlement_head_refresh_failure_cannot_create_a_lease():
    class Store:
        lease_calls = 0

        def has_pending_settlement(self):
            return True

        def lease_settlement_cohort(self, *, current_block):
            self.lease_calls += 1
            return None

    def unavailable_head():
        raise RuntimeError("finalized head unavailable")

    store = Store()
    with pytest.raises(RuntimeError, match="finalized head unavailable"):
        loop._settle_pending(
            store,
            current_block=BLOCK,
            finalized_block_provider=unavailable_head,
        )
    assert store.lease_calls == 0
