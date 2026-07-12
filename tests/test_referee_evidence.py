from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.check_referee_evidence import (
    EvidenceError,
    classify,
    diff_facts,
    load_json,
    validate_contract_document,
    validate_record,
    validate_repository,
    validate_scope,
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
    with pytest.raises(EvidenceError, match="in-place"):
        validate_scope(contract, {"optima/eval/qualification.py": ("A", 1, 0)})
    with pytest.raises(EvidenceError, match="budget"):
        validate_scope(
            contract,
            {
                "optima/eval/scoring.py": ("M", 1, 0),
                "optima/eval/qualification.py": ("A", 3200, 0),
            },
        )


def test_duplicate_json_keys_fail(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(EvidenceError, match="duplicate JSON key"):
        load_json(path)
