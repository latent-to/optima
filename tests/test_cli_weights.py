from __future__ import annotations

import argparse

import pytest

import optima.cli as cli
from optima import chain
from optima.chain.intake import (
    FinalizedIntakeStore,
    IntakeError,
    IntakeScope,
    SQLiteWeightPublicationJournal,
)
from optima.chain.weights import (
    WeightProjection,
    WeightPublicationError,
    WeightPublicationRecord,
)
from optima.economics import EmissionsPolicyManifest
from optima.stack_identity import canonical_digest, sha256_hex


SCOPE = IntakeScope("0x" + "0" * 64, 307)
POLICY = EmissionsPolicyManifest(100, 20, 100_000)


def _h(label: str) -> str:
    return sha256_hex(label.encode())


def _view(block: int) -> chain.MetagraphView:
    return chain.MetagraphView(
        307,
        block,
        "0x" + f"{block:064x}",
        [0, 1],
        ["validator", "miner"],
        [True, True],
        [10, 0],
    )


def _projection() -> WeightProjection:
    bound = _view(10)
    metagraph_digest = canonical_digest(
        "optima.economics.metagraph-membership",
        {
            "block": bound.block,
            "block_hash": bound.block_hash,
            "chain_scope_digest": SCOPE.digest,
            "members": [
                {"hotkey": hotkey, "uid": uid}
                for uid, hotkey in zip(bound.uids, bound.hotkeys, strict=True)
            ],
        },
    )
    return WeightProjection(
        SCOPE.digest,
        307,
        "validator",
        POLICY.digest,
        _h("settlement"),
        _h("evaluation"),
        metagraph_digest,
        (_h("arena-state"),),
        1,
        10,
        1,
        (_h("off-pod-evidence"),),
        (("miner", 1_000_000),),
    )


def _seed_pending(path) -> tuple[WeightProjection, WeightPublicationRecord]:
    projection = _projection()
    pending = WeightPublicationRecord(
        projection.digest,
        "pending",
        submit_block=10,
        retry_after_block=30,
        reason="sdk_result_unconfirmed",
    )
    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        SQLiteWeightPublicationJournal(store, projection).compare_and_swap(
            None, pending
        )
    return projection, pending


def _seed_held(path) -> tuple[WeightProjection, WeightPublicationRecord]:
    projection = _projection()
    held = WeightPublicationRecord(
        projection.digest,
        "held",
        submit_block=10,
        retry_after_block=30,
        reason="publication_readback_deadline_expired",
    )
    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        SQLiteWeightPublicationJournal(store, projection).compare_and_swap(None, held)
    return projection, held


def _args(path, **updates) -> argparse.Namespace:
    values = {
        "intake_db": str(path),
        "netuid": 307,
        "network": "mock",
        "wallet": "must-not-load",
        "hotkey": "must-not-load",
        "validator_hotkey": "validator",
        "half_life_blocks": 100,
        "discovery_lifetime_blocks": 20,
        "discovery_pool_ppm": 100_000,
        "refresh_blocks": 20,
        "release_hold": "",
        "burn_hotkey": "",
        "dry_run": False,
        "reconcile_only": True,
    }
    values.update(updates)
    return argparse.Namespace(**values)


class _Subtensor:
    def get_block_hash(self, block: int) -> str:
        assert block == 0
        return SCOPE.genesis_hash


def _install_chain_readback(monkeypatch) -> None:
    monkeypatch.setattr(chain, "connect", lambda _network: _Subtensor())
    monkeypatch.setattr(
        chain,
        "fetch_metagraph",
        lambda _subtensor, _netuid, *, block=None: _view(
            11 if block is None else block
        ),
    )
    monkeypatch.setattr(
        chain,
        "read_validator_weight_snapshot",
        lambda *_args, **_kwargs: chain.ValidatorWeightSnapshot(
            {"miner": 1.0}, 10
        ),
    )


def test_reconcile_only_cli_uses_retained_head_without_reopening_evidence(
    tmp_path, monkeypatch
):
    path = tmp_path / "private" / "intake.sqlite3"
    projection, pending = _seed_pending(path)
    _install_chain_readback(monkeypatch)

    def forbidden_fresh_projection(*_args, **_kwargs):
        raise AssertionError("reconcile-only reopened current settlement evidence")

    monkeypatch.setattr(
        FinalizedIntakeStore,
        "build_weight_projection",
        forbidden_fresh_projection,
    )

    assert cli.cmd_set_weights(_args(path)) == 0

    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        journal = SQLiteWeightPublicationJournal.reopen_from_head(store)
        assert journal.projection == projection
        confirmed = journal.load()
        assert confirmed is not None
        assert confirmed.status == "confirmed"
        assert confirmed.prior_record_digest == pending.digest


def test_reconcile_only_cli_reports_historical_confirmation_that_needs_refresh(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "private" / "intake.sqlite3"
    _projection_row, pending = _seed_pending(path)
    monkeypatch.setattr(chain, "connect", lambda _network: _Subtensor())
    monkeypatch.setattr(
        chain,
        "fetch_metagraph",
        lambda _subtensor, _netuid, *, block=None: _view(
            31 if block is None else block
        ),
    )
    monkeypatch.setattr(
        chain,
        "read_validator_weight_snapshot",
        lambda *_args, **_kwargs: chain.ValidatorWeightSnapshot(
            {"miner": 1.0}, 10
        ),
    )

    assert cli.cmd_set_weights(_args(path)) == 3
    output = capsys.readouterr().out
    assert "status=confirmed" in output
    assert "refresh_due=True" in output

    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        confirmed = SQLiteWeightPublicationJournal.reopen_from_head(store).load()
        assert confirmed is not None
        assert confirmed.status == "confirmed"
        assert confirmed.prior_record_digest == pending.digest


def test_release_hold_reopens_retained_head_without_off_pod_evidence(
    tmp_path, monkeypatch
):
    path = tmp_path / "private" / "intake.sqlite3"
    projection, held = _seed_held(path)
    monkeypatch.setattr(chain, "connect", lambda _network: _Subtensor())

    def forbidden_fresh_projection(*_args, **_kwargs):
        raise AssertionError("release-hold reopened current settlement evidence")

    monkeypatch.setattr(
        FinalizedIntakeStore,
        "build_weight_projection",
        forbidden_fresh_projection,
    )

    assert cli.cmd_set_weights(
        _args(
            path,
            reconcile_only=False,
            release_hold="late reveal reviewed",
        )
    ) == 0

    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        journal = SQLiteWeightPublicationJournal.reopen_from_head(store)
        assert journal.projection == projection
        released = journal.load()
        assert released is not None
        assert released.status == "released"
        assert released.prior_record_digest == held.digest


def test_release_hold_authority_mismatch_does_not_mutate_head(
    tmp_path, monkeypatch
):
    path = tmp_path / "private" / "intake.sqlite3"
    projection, held = _seed_held(path)
    monkeypatch.setattr(chain, "connect", lambda _network: _Subtensor())

    with pytest.raises(WeightPublicationError, match="public validator hotkey"):
        cli.cmd_set_weights(
            _args(
                path,
                reconcile_only=False,
                release_hold="must not land",
                validator_hotkey="other",
            )
        )

    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        journal = SQLiteWeightPublicationJournal.reopen_from_head(store)
        assert journal.projection == projection
        assert journal.load() == held


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"validator_hotkey": "other"}, "public validator hotkey"),
        ({"half_life_blocks": 101}, "emissions policy"),
    ],
)
def test_reconcile_only_cli_rejects_public_authority_mismatch_before_readback(
    tmp_path, monkeypatch, updates, message
):
    path = tmp_path / "private" / "intake.sqlite3"
    projection, pending = _seed_pending(path)
    monkeypatch.setattr(chain, "connect", lambda _network: _Subtensor())

    def forbidden_readback(*_args, **_kwargs):
        raise AssertionError("mismatched retained authority reached chain readback")

    monkeypatch.setattr(chain, "fetch_metagraph", forbidden_readback)

    with pytest.raises(WeightPublicationError, match=message):
        cli.cmd_set_weights(_args(path, **updates))

    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        journal = SQLiteWeightPublicationJournal.reopen_from_head(store)
        assert journal.projection == projection
        assert journal.load() == pending


def test_burn_hotkey_cli_dry_run_projects_the_full_pool_pre_crown(
    tmp_path, monkeypatch, capsys
):
    import sys
    import types

    path = tmp_path / "private" / "intake.sqlite3"
    with FinalizedIntakeStore(path, scope=SCOPE):
        pass  # an empty all-uncrowned store is the entire precondition
    _install_chain_readback(monkeypatch)

    class _Hotkey:
        ss58_address = "validator"

    class _PublicWallet:
        def __init__(self, name, hotkey):
            assert name == "must-not-load"
            self.hotkey = _Hotkey()

    monkeypatch.setitem(
        sys.modules, "bittensor", types.SimpleNamespace(Wallet=_PublicWallet)
    )

    assert cli.cmd_set_weights(
        _args(
            path,
            reconcile_only=False,
            dry_run=True,
            validator_hotkey="",
            burn_hotkey="miner",
        )
    ) == 0
    out = capsys.readouterr().out
    assert "burn projection: full pool -> uid 1 hotkey miner" in out
    assert "status=dry_run" in out
    assert "submitted=False" in out

    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        with pytest.raises(IntakeError, match="no retained head"):
            SQLiteWeightPublicationJournal.reopen_from_head(store)


def test_burn_hotkey_cli_refuses_head_only_combinations(tmp_path):
    path = tmp_path / "private" / "intake.sqlite3"
    with pytest.raises(SystemExit, match="burn-hotkey"):
        cli.cmd_set_weights(_args(path, burn_hotkey="miner"))
    with pytest.raises(SystemExit, match="burn-hotkey"):
        cli.cmd_set_weights(
            _args(
                path,
                reconcile_only=False,
                release_hold="operator request",
                burn_hotkey="miner",
            )
        )


def test_burn_hotkey_cli_publishes_real_weights_without_a_crown(
    tmp_path, monkeypatch, capsys
):
    import sys
    import types

    path = tmp_path / "private" / "intake.sqlite3"
    with FinalizedIntakeStore(path, scope=SCOPE):
        pass
    monkeypatch.setattr(chain, "connect", lambda _network: _Subtensor())
    monkeypatch.setattr(
        chain,
        "fetch_metagraph",
        lambda _subtensor, _netuid, *, block=None: _view(
            11 if block is None else block
        ),
    )
    snapshots = [
        chain.ValidatorWeightSnapshot({"validator": 1.0}, 5),
        chain.ValidatorWeightSnapshot({"miner": 1.0}, 11),
    ]
    monkeypatch.setattr(
        chain,
        "read_validator_weight_snapshot",
        lambda *_args, **_kwargs: snapshots.pop(0),
    )
    submissions = []

    def _set_weights(_subtensor, wallet, netuid, weights, **kwargs):
        submissions.append((wallet, netuid, dict(weights), kwargs))
        return {"submitted": True}

    monkeypatch.setattr(chain, "set_weights", _set_weights)

    class _Hotkey:
        ss58_address = "validator"

    class _SignerWallet:
        def __init__(self, name, hotkey):
            self.hotkey = _Hotkey()

    monkeypatch.setitem(
        sys.modules, "bittensor", types.SimpleNamespace(Wallet=_SignerWallet)
    )

    assert cli.cmd_set_weights(
        _args(
            path,
            reconcile_only=False,
            validator_hotkey="",
            burn_hotkey="miner",
        )
    ) == 0
    out = capsys.readouterr().out
    assert "status=confirmed" in out
    assert "submitted=True" in out
    assert len(submissions) == 1
    assert submissions[0][2] == {"miner": 1.0}
    assert submissions[0][3]["dry_run"] is False

    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        journal = SQLiteWeightPublicationJournal.reopen_from_head(store)
        assert journal.projection.weights == {"miner": 1.0}
        assert journal.projection.crown_count == 0
        confirmed = journal.load()
        assert confirmed is not None
        assert confirmed.status == "confirmed"
