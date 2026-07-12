from __future__ import annotations

import argparse

import pytest

import optima.cli as cli


def test_chain_validate_refuses_implicit_fake_grading(monkeypatch):
    args = argparse.Namespace(intake_only=False)
    with pytest.raises(SystemExit, match="requires --intake-only"):
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
    assert not {
        "--eval-cmd", "--eval-device", "--eval-timeout", "--margin",
        "--wallet", "--hotkey", "--dry-run-weights",
    } & options
    assert "chain-validate" in source
