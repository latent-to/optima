from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

import optima.cli as cli


def test_chain_validate_refuses_implicit_fake_grading(monkeypatch):
    args = argparse.Namespace(intake_only=False)
    with pytest.raises(SystemExit, match="requires --intake-only or"):
        cli.cmd_chain_validate(args)


def test_chain_validate_intake_path_has_no_wallet_or_weight_arguments():
    source = cli.build_parser().format_help()
    # Global help routes commands rather than rendering subparser flags; inspect the
    # parser action directly without executing any chain code.
    parser = cli.build_parser()
    subparsers = next(
        action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    chain_validate = subparsers.choices["chain-validate"]
    options = {
        option
        for action in chain_validate._actions
        for option in action.option_strings
    }
    assert "--intake-only" in options
    assert "--arena-id" in options
    assert not {
        "--eval-cmd", "--eval-device", "--eval-timeout", "--margin",
        "--wallet", "--hotkey", "--dry-run-weights",
    } & options
    assert "chain-validate" in source


def test_chain_activation_cli_is_wallet_free_and_forwards_exact_authority(
    monkeypatch, capsys
):
    parser = cli.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    command = subparsers.choices["chain-activate-incentives"]
    options = {
        option for action in command._actions for option in action.option_strings
    }
    assert {
        "--intake-db",
        "--netuid",
        "--network",
        "--core-policy",
        "--composition-policy",
        "--approval",
        "--expected-approval-digest",
    } <= options
    assert not {"--wallet", "--hotkey", "--password", "--dry-run"} & options

    import optima.chain.incentive_activation as activation_module

    observed = {}
    result = SimpleNamespace(
        digest="f" * 64,
        to_dict=lambda: {"activation_digest": "e" * 64},
    )

    def execute(**kwargs):
        observed.update(kwargs)
        return result

    monkeypatch.setattr(
        activation_module,
        "execute_selected_incentive_activation",
        execute,
    )
    args = argparse.Namespace(
        network="mock",
        netuid=307,
        intake_db="intake.sqlite3",
        core_policy="core.json",
        composition_policy="composition.json",
        approval="approval.json",
        expected_approval_digest="a" * 64,
    )
    assert cli.cmd_chain_activate_incentives(args) == 0
    assert observed["network"] == "mock"
    assert observed["netuid"] == 307
    assert observed["expected_approval_digest"] == "a" * 64
    assert callable(observed["connect"])
    assert callable(observed["read_finalized_head"])
    assert callable(observed["fetch_metagraph"])
    output = capsys.readouterr().out
    assert '"result_digest": "' + "f" * 64 + '"' in output
