from __future__ import annotations

import os
from pathlib import Path

import pytest

import optima.chain.validator_loop as loop
from optima.bundle_hash import content_hash
from optima.chain import FinalizedRevealSnapshot, RevealedCommitment
from optima.chain.intake import FinalizedIntakeStore, IntakeScope
from optima.chain.payload import encode_payload
from optima.eval.evidence_store import EvidenceArtifactRef
from optima.eval.qualification import QualificationDecision
from optima.eval.qualification_intake import (
    QualificationIntakeBatch, QualificationIntakeOutcome,
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


def test_live_loop_calls_batch_qualification_and_retains_outcome(
    tmp_path, monkeypatch
):
    source = _bundle(
        tmp_path / "source",
        "def silu_and_mul(x, out):\n    out.copy_(x)\n",
    )
    digest = content_hash(source)
    snapshot = _snapshot([("miner", encode_payload(digest, "https://example.com/a"))])

    class FakeFactory:
        def __init__(self, reservations):
            self.manifest = type("Manifest", (), {
                "reservations": reservations,
                "digest": "a" * 64,
                "to_dict": lambda self: {"digest": self.digest},
            })()

    monkeypatch.setattr(loop, "QualificationPlanFactory", FakeFactory)
    calls = []

    def planner(_reservations, _publications, authority_rows):
        return loop.QualificationWork(
            FakeFactory(authority_rows), object(), lambda *_: None, lambda **_: None, 30.0
        )

    def qualify(factory, **_kwargs):
        calls.append(factory.manifest.digest)
        authority = factory.manifest.reservations[0]
        outcome = QualificationIntakeOutcome(
            authority.reservation_digest,
            authority.selected_delta_digest,
            factory.manifest.digest,
            QualificationDecision.PASS,
            "qualified",
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
        qualification_planner=planner,
    )
    assert calls == ["a" * 64]
    assert set(result.decisions.values()) == {"PASS"}
    with FinalizedIntakeStore(options["intake_db"], scope=SCOPE) as store:
        row = store.all()[0]
        assert row.status == "qualified" and row.decision == "PASS"
        assert store.qualification_dispositions(row.reservation_id)[0]["decision"] == "PASS"


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
