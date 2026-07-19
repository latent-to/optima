from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace

import optima.cli as cli
from optima import chain
from optima.chain.intake import FinalizedIntakeStore
from tests.test_chain_intake import SCOPE, _store
from tests.test_incentive_composition_store import (
    _activate_selected,
    _default_family,
)


def _args(path, **updates) -> argparse.Namespace:
    values = {
        "intake_db": str(path),
        "netuid": SCOPE.netuid,
        "network": "mock",
        "wallet": "default",
        "hotkey": "default",
        "validator_hotkey": "",
        "refresh_blocks": 20,
        "release_hold": "",
        "dry_run": False,
        "reconcile_only": False,
    }
    values.update(updates)
    return argparse.Namespace(**values)


class _Subtensor:
    def get_block_hash(self, block: int) -> str:
        assert block == 0
        return SCOPE.genesis_hash


def _install_chain(monkeypatch, *, block: int, apply: bool):
    state = {"block": block, "weights": {}, "last_update": 0, "submits": 0}
    subtensor = _Subtensor()
    monkeypatch.setattr(chain, "connect", lambda _network: subtensor)
    monkeypatch.setattr(
        chain,
        "read_finalized_head",
        lambda _subtensor: (
            state["block"],
            "0x" + f"{state['block']:064x}",
        ),
    )

    def metagraph(_subtensor, netuid, *, block=None):
        height = state["block"] if block is None else block
        return chain.MetagraphView(
            netuid,
            height,
            "0x" + f"{height:064x}",
            [0, 1],
            ["validator", "reserve"],
            [True, True],
            [state["last_update"], 0],
        )

    monkeypatch.setattr(chain, "fetch_metagraph", metagraph)
    monkeypatch.setattr(
        chain,
        "read_validator_weight_snapshot",
        lambda *_args, **_kwargs: chain.ValidatorWeightSnapshot(
            dict(state["weights"]), state["last_update"]
        ),
    )

    def set_weights(
        _subtensor,
        _wallet,
        _netuid,
        weights,
        **_kwargs,
    ):
        state["submits"] += 1
        if apply:
            state["weights"] = dict(weights)
            state["last_update"] = state["block"]
        return {"submitted": True}

    monkeypatch.setattr(chain, "set_weights", set_weights)
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator"))
    monkeypatch.setitem(
        sys.modules,
        "bittensor",
        SimpleNamespace(Wallet=lambda **_kwargs: wallet),
    )
    return state


def _activated_store(tmp_path, *, cursor: int) -> str:
    with _store(tmp_path) as store:
        _activate_selected(store, _default_family())
        store.reserve_finalized(
            (),
            finalized_block=cursor,
            finalized_block_hash="0x" + f"{cursor:064x}",
        )
        return str(store.path)


def test_set_debt_weights_publishes_confirms_and_debits_one_boundary(
    tmp_path, monkeypatch, capsys
) -> None:
    path = _activated_store(tmp_path, cursor=7_210)
    state = _install_chain(monkeypatch, block=7_210, apply=True)

    assert cli.cmd_set_debt_weights(_args(path)) == 0
    output = capsys.readouterr().out
    assert "status=confirmed" in output
    assert "closed=True" in output
    assert state["weights"] == {"reserve": 1.0}
    assert state["submits"] == 1

    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        epochs = store.incentive_composition_reward_epochs()
        assert len(epochs) == 1 and epochs[0].effective_block == 7_210
        confirmation = store.confirmed_debt_weight_publication(
            epochs[0].publication_record_digest
        )
        assert confirmation.projection_digest == epochs[0].projection.digest
        assert confirmation.readback.block == 7_210


def test_set_debt_weights_restarts_after_intake_catches_confirmed_chain(
    tmp_path, monkeypatch, capsys
) -> None:
    path = _activated_store(tmp_path, cursor=7_210)
    state = _install_chain(monkeypatch, block=7_211, apply=True)

    assert cli.cmd_set_debt_weights(_args(path)) == 3
    first = capsys.readouterr().out
    assert "status=confirmed" in first
    assert "closed=False" in first
    assert "awaiting_intake=True" in first
    assert state["submits"] == 1
    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        assert store.incentive_composition_reward_epochs() == ()
        store.reserve_finalized(
            (),
            finalized_block=7_211,
            finalized_block_hash="0x" + f"{7211:064x}",
        )

    assert cli.cmd_set_debt_weights(
        _args(
            path,
            reconcile_only=True,
            validator_hotkey="validator",
        )
    ) == 0
    second = capsys.readouterr().out
    assert "status=confirmed" in second
    assert "closed=True" in second
    assert state["submits"] == 1
    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        assert len(store.incentive_composition_reward_epochs()) == 1


def test_set_debt_weights_not_due_retains_no_publication(
    tmp_path, monkeypatch, capsys
) -> None:
    path = _activated_store(tmp_path, cursor=100)
    state = _install_chain(monkeypatch, block=100, apply=True)

    assert cli.cmd_set_debt_weights(_args(path)) == 0
    assert "status=not_due" in capsys.readouterr().out
    assert state["submits"] == 0
    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        assert store.incentive_composition_reward_epochs() == ()
        assert store._db.execute(
            "SELECT COUNT(*) AS value FROM debt_weight_publication_journal"
        ).fetchone()["value"] == 0


def test_set_debt_weights_pending_requires_operator_followup(
    tmp_path, monkeypatch, capsys
) -> None:
    path = _activated_store(tmp_path, cursor=7_210)
    state = _install_chain(monkeypatch, block=7_210, apply=False)

    assert cli.cmd_set_debt_weights(_args(path)) == 3
    output = capsys.readouterr().out
    assert "status=pending" in output
    assert "closed=False" in output
    assert state["submits"] == 1

    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        assert store.incentive_composition_reward_epochs() == ()
        journal = store.debt_weight_publication_journal()
        assert journal.load().status == "pending"


def test_set_debt_weights_dry_run_prints_exact_binding_without_fence(
    tmp_path, monkeypatch, capsys
) -> None:
    path = _activated_store(tmp_path, cursor=7_210)
    _install_chain(monkeypatch, block=7_210, apply=False)

    assert cli.cmd_set_debt_weights(_args(path, dry_run=True)) == 0
    lines = capsys.readouterr().out.splitlines()
    reviewed = json.loads(lines[0])["debt_weight_publication_binding"]
    assert reviewed["economic_projection"]["effective_block"] == 7_210
    assert reviewed["weight_projection"]["weights_ppm"] == [
        ["reserve", 1_000_000]
    ]
    assert "status=dry_run" in lines[-1]

    with FinalizedIntakeStore(path, scope=SCOPE) as store:
        assert store.incentive_composition_reward_epochs() == ()
        assert store._db.execute(
            "SELECT COUNT(*) AS value FROM debt_weight_publication_journal"
        ).fetchone()["value"] == 0
