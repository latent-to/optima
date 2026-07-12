from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.check_referee_evidence import (
    EvidenceError,
    classify,
    diff_facts,
    load_json,
    validate_authority_boundary,
    validate_contract_document,
    validate_record,
    validate_repository,
    validate_scope,
    validate_v2_schema_contract,
)


ROOT = Path(__file__).parents[1]
EVIDENCE = ROOT / "evidence/referee-hardening"


def test_historical_records_recompute_from_git() -> None:
    validate_repository(ROOT, records_only=True)


def test_exact_historical_classification() -> None:
    counts, _ = diff_facts(
        ROOT,
        "38ddaed6947228f0777a85e67c2160490353ea3b",
        "e398fa01d9e3f0c999e7ef18e410f9778cd01a89",
    )
    assert counts["production"] == {"added": 791, "deleted": 3}
    assert counts["test"] == {"added": 910, "deleted": 0}
    assert classify("optima/eval/seccomp_moby_v0_2_1.json") == "vendor"


def test_pr43_preserves_invalid_exact_head_claim() -> None:
    record = load_json(EVIDENCE / "records/pr-0043.json")
    claim = next(item for item in record["claims"] if item["id"] == "tests.full-exact-head")
    assert claim["status"] == "invalidated"
    assert "oci_prebuild" in claim["statement"]
    broken = copy.deepcopy(record)
    claim = next(item for item in broken["claims"] if item["id"] == "tests.full-exact-head")
    claim["status"] = "reported_unretained"
    with pytest.raises(EvidenceError, match="PR 43"):
        validate_record(broken, ROOT, "broken")


def test_contract_rejects_scope_growth_and_budget_exemption() -> None:
    contract = load_json(EVIDENCE / "contracts/pr4a.json")
    validate_contract_document(contract)
    widened = copy.deepcopy(contract)
    widened["budget"]["exemption_policy"] = "written"
    with pytest.raises(EvidenceError, match="budget"):
        validate_contract_document(widened)
    changes = {
        "optima/eval/scoring.py": ("M", 10, 4),
        "optima/eval/qualification.py": ("A", 100, 0),
        "optima/chain/validator_loop.py": ("M", 1, 0),
    }
    with pytest.raises(EvidenceError, match="outside frozen"):
        validate_scope(contract, changes)


def test_contract_requires_in_place_scoring_and_caps_lines() -> None:
    contract = load_json(EVIDENCE / "contracts/pr4a.json")
    with pytest.raises(EvidenceError, match="required modify"):
        validate_scope(contract, {"optima/eval/qualification.py": ("A", 1, 0)})
    with pytest.raises(EvidenceError, match="budget"):
        validate_scope(
            contract,
            {
                "optima/eval/scoring.py": ("M", 1, 0),
                "optima/eval/qualification.py": ("A", 3200, 0),
            },
        )


def test_pr47_retains_exact_github_check_receipt() -> None:
    record = load_json(EVIDENCE / "records/pr-0047.json")
    validate_record(record, ROOT, "pr47")
    artifact = next(item for item in record["artifacts"] if item["id"] == "github.referee-evidence")
    assert artifact["availability"] == "repository"
    receipt = load_json(ROOT / artifact["locator"])
    assert receipt["pull_request"] == 47
    assert receipt["total_count"] == 4
    assert {item["conclusion"] for item in receipt["check_runs"]} == {"success"}
    assert {item["head_sha"] for item in receipt["check_runs"]} == {
        "e60fb8561094b6a325107bb838feec4ad35743f7"
    }


def test_pr4b_contract_closes_scope_budget_and_required_replacements() -> None:
    contract = load_json(EVIDENCE / "contracts/pr4b.json")
    validate_contract_document(contract)
    changes = {
        item["path"]: ({"add": "A", "modify": "M"}[item["change"]], 1, 0)
        for item in contract["required_in_place"]
    }
    validate_scope(contract, changes)

    widened = copy.deepcopy(contract)
    widened["authority_boundary"]["roots"].append("optima.eval.new_authority")
    with pytest.raises(EvidenceError, match="authority boundary"):
        validate_contract_document(widened)

    outside = dict(changes)
    outside["optima/eval/new_grader.py"] = ("A", 1, 0)
    with pytest.raises(EvidenceError, match="outside frozen pr4b"):
        validate_scope(contract, outside)

    over_budget = dict(changes)
    over_budget["optima/eval/reference_protocol.py"] = ("A", 3301, 0)
    with pytest.raises(EvidenceError, match="pr4b line budget"):
        validate_scope(contract, over_budget)

    schema = load_json(EVIDENCE / "schema-v2.json")
    validate_v2_schema_contract(schema, contract)
    stale = copy.deepcopy(schema)
    scope = next(
        row for row in stale["oneOf"]
        if row.get("properties", {}).get("record_type", {}).get("const")
        == "scope_contract"
    )
    scope["properties"]["allowed_paths"]["const"]["production"].append(
        "optima/eval/qualification_runner.py"
    )
    with pytest.raises(EvidenceError, match="scope constant differs"):
        validate_v2_schema_contract(stale, contract)


def test_authority_boundary_walks_transitive_and_relative_repo_imports(tmp_path: Path) -> None:
    eval_dir = tmp_path / "optima/eval"
    eval_dir.mkdir(parents=True)
    (tmp_path / "optima/__init__.py").write_text("")
    (eval_dir / "__init__.py").write_text("")
    (eval_dir / "root.py").write_text("from .bridge import value\n")
    (eval_dir / "bridge.py").write_text("from optima.eval import _launch\nvalue = 1\n")
    (eval_dir / "_launch.py").write_text("value = 1\n")
    contract = {
        "authority_boundary": {
            "forbidden_modules": ["optima.eval._launch"],
            "roots": ["optima.eval.root"],
        }
    }
    with pytest.raises(EvidenceError, match=r"root.*bridge.*_launch"):
        validate_authority_boundary(tmp_path, contract)

    (eval_dir / "bridge.py").write_text("value = 1\n")
    validate_authority_boundary(tmp_path, contract)


def test_authority_boundary_rejects_absent_declared_root(tmp_path: Path) -> None:
    (tmp_path / "optima").mkdir()
    contract = {
        "authority_boundary": {
            "forbidden_modules": ["optima.eval._launch"],
            "roots": ["optima.eval.missing"],
        }
    }
    with pytest.raises(EvidenceError, match="authority root is absent"):
        validate_authority_boundary(tmp_path, contract)


def test_authority_boundary_follows_and_bounds_dynamic_imports(tmp_path: Path) -> None:
    eval_dir = tmp_path / "optima/eval"
    eval_dir.mkdir(parents=True)
    (tmp_path / "optima/__init__.py").write_text("")
    (eval_dir / "__init__.py").write_text("")
    (eval_dir / "root.py").write_text(
        "from importlib import import_module\nimport_module('optima.eval._launch')\n"
    )
    (eval_dir / "_launch.py").write_text("value = 1\n")
    contract = {
        "authority_boundary": {
            "forbidden_modules": ["optima.eval._launch"],
            "roots": ["optima.eval.root"],
        }
    }
    with pytest.raises(EvidenceError, match=r"root.*_launch"):
        validate_authority_boundary(tmp_path, contract)
    (eval_dir / "root.py").write_text(
        "from importlib import import_module\nname = 'optima.eval._launch'\n"
        "import_module(name)\n"
    )
    with pytest.raises(EvidenceError, match="unresolved dynamic import"):
        validate_authority_boundary(tmp_path, contract)

    (eval_dir / "root.py").write_text(
        "from importlib import import_module\nimport_module('._launch', __package__)\n"
    )
    with pytest.raises(EvidenceError, match=r"root.*_launch"):
        validate_authority_boundary(tmp_path, contract)

    (eval_dir / "root.py").write_text(
        "__import__('optima.eval', fromlist=('_launch',))\n"
    )
    with pytest.raises(EvidenceError, match=r"root.*_launch"):
        validate_authority_boundary(tmp_path, contract)

    for source in (
        "from builtins import __import__ as imp\n"
        "imp('optima.eval', fromlist=('_launch',))\n",
        "import builtins as b\n"
        "b.__import__('optima.eval', fromlist=('_launch',))\n",
    ):
        (eval_dir / "root.py").write_text(source)
        with pytest.raises(EvidenceError, match=r"root.*_launch"):
            validate_authority_boundary(tmp_path, contract)


def test_duplicate_json_keys_fail(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(EvidenceError, match="duplicate JSON key"):
        load_json(path)
