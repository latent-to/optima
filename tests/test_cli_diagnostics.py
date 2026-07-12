"""Direct evaluate/bench commands are development diagnostics, not settlement."""

from __future__ import annotations

import inspect

import pytest

from optima import cli


@pytest.mark.parametrize("command", ("evaluate", "bench"))
def test_direct_diagnostics_have_no_ledger_authority(command):
    parser = cli.build_parser()
    args = parser.parse_args([command, "bundle", "--model", "model"])

    assert args.func is getattr(cli, f"cmd_{command}")
    assert not hasattr(args, "ledger")
    assert not hasattr(args, "hotkey")
    assert not hasattr(args, "round")

    source = inspect.getsource(args.func)
    assert "record_score" not in source
    assert "Ledger" not in source
    assert "diagnostic score" in source
    assert "crownable speedup" not in source


@pytest.mark.parametrize("command", ("evaluate", "bench"))
@pytest.mark.parametrize(
    "legacy_option,value",
    (("--ledger", "ledger.json"), ("--hotkey", "miner"), ("--round", "7")),
)
def test_direct_diagnostics_reject_legacy_settlement_options(
    command, legacy_option, value
):
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [command, "bundle", "--model", "model", legacy_option, value]
        )
    assert exc_info.value.code == 2


def test_chain_simulation_keeps_its_own_ledger_options():
    args = cli.build_parser().parse_args(
        ["commit", "bundle", "--hotkey", "miner", "--salt", "secret"]
    )
    assert args.ledger == "optima_ledger.json"
    assert args.hotkey == "miner"
    assert args.round == 0


def test_root_help_labels_direct_commands_as_diagnostics():
    help_text = cli.build_parser().format_help()
    assert "evaluate" in help_text and "development diagnostic" in help_text
    assert "bench" in help_text
