#!/usr/bin/env python3
"""Validate referee evidence records and the frozen change contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

SHA256 = re.compile(r"[0-9a-f]{64}\Z")
GIT_OID = re.compile(r"[0-9a-f]{40}\Z")
IDENT = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
CATEGORIES = ("documentation", "evidence", "production", "test", "tooling", "vendor")
RECORD_KEYS = {
    "architectural_unit", "artifacts", "claims", "classified_diff", "exit_criteria",
    "github_pr", "identity", "nonclaims", "record_type", "reviews", "schema_version",
}
CONTRACT_KEYS = {
    "allowed_paths", "architectural_unit", "base_commit", "budget", "contract_id",
    "record_type", "required_in_place", "schema_version",
}


class EvidenceError(ValueError):
    pass


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read {path}: {exc}") from exc
    if type(value) is not dict:
        raise EvidenceError(f"{path} must contain one object")
    return value


def _keys(value: dict[str, Any], expected: set[str], where: str) -> None:
    if set(value) != expected:
        raise EvidenceError(f"{where} keys differ: {sorted(set(value) ^ expected)}")


def _sha(value: Any, where: str) -> str:
    if type(value) is not str or not SHA256.fullmatch(value):
        raise EvidenceError(f"{where} must be a lowercase SHA-256")
    return value


def _git_oid(value: Any, where: str) -> str:
    if type(value) is not str or not GIT_OID.fullmatch(value):
        raise EvidenceError(f"{where} must be a full lowercase Git object ID")
    return value


def _ident(value: Any, where: str) -> str:
    if type(value) is not str or not IDENT.fullmatch(value):
        raise EvidenceError(f"{where} must be a stable identifier")
    return value


def git(root: Path, *args: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=root, check=False, capture_output=True, text=text
    )
    if result.returncode:
        error = result.stderr.strip() if text else result.stderr.decode(errors="replace").strip()
        raise EvidenceError(f"git {' '.join(args)} failed: {error}")
    return result.stdout.strip() if text else result.stdout


def classify(path: str) -> str:
    if path.startswith("evidence/"):
        return "evidence"
    if path.startswith("tests/"):
        return "test"
    if path.startswith("docs/") or path in {"README.md", "AGENTS.md"}:
        return "documentation"
    if path.startswith(("scripts/", ".github/workflows/")):
        return "tooling"
    if (
        path.startswith(("LICENSES/", "optima/arena_assets/"))
        or path in {
            ".gitattributes", "NOTICE", "optima/eval/seccomp_moby_v0_2_1.json",
            "optima/vendor_provenance.json",
        }
    ):
        return "vendor"
    return "production"


def diff_facts(root: Path, base: str, head: str) -> tuple[dict[str, dict[str, int]], dict[str, str]]:
    counts = {name: {"added": 0, "deleted": 0} for name in CATEGORIES}
    statuses: dict[str, str] = {}
    for line in str(git(root, "diff", "--numstat", "--no-renames", f"{base}..{head}")).splitlines():
        added, deleted, path = line.split("\t", 2)
        if not added.isdigit() or not deleted.isdigit():
            raise EvidenceError(f"binary diff is not admissible: {path}")
        category = classify(path)
        counts[category]["added"] += int(added)
        counts[category]["deleted"] += int(deleted)
    for line in str(git(root, "diff", "--name-status", "--no-renames", f"{base}..{head}")).splitlines():
        status, path = line.split("\t", 1)
        statuses[path] = status
    return counts, statuses


def _validate_artifact(item: dict[str, Any], root: Path, commit: str, where: str) -> None:
    _keys(item, {"availability", "id", "kind", "locator", "reason", "sha256", "size_bytes"}, where)
    _ident(item["id"], f"{where}.id")
    _ident(item["kind"], f"{where}.kind")
    availability = item["availability"]
    if item["locator"] is not None and type(item["locator"]) is not str:
        raise EvidenceError(f"{where}.locator must be a string or null")
    if item["reason"] is not None and (type(item["reason"]) is not str or not item["reason"]):
        raise EvidenceError(f"{where}.reason must be a nonempty string or null")
    if availability not in {"repository", "digest_only", "unavailable"}:
        raise EvidenceError(f"{where}.availability is invalid")
    if availability == "unavailable":
        if item["sha256"] is not None or item["size_bytes"] is not None or not item["reason"]:
            raise EvidenceError(f"{where} unavailable artifact must state absence without a digest")
        return
    digest = _sha(item["sha256"], f"{where}.sha256")
    if availability == "digest_only":
        if item["size_bytes"] is not None or not item["reason"]:
            raise EvidenceError(f"{where} digest-only artifact must state why bytes are absent")
        return
    if type(item["locator"]) is not str or item["locator"].startswith("/"):
        raise EvidenceError(f"{where} repository locator must be relative")
    raw = bytes(git(root, "show", f"{commit}:{item['locator']}", text=False))
    if hashlib.sha256(raw).hexdigest() != digest or len(raw) != item["size_bytes"]:
        raise EvidenceError(f"{where} repository artifact bytes do not match")
    if item["reason"] is not None:
        raise EvidenceError(f"{where} retained artifact cannot have an absence reason")


def validate_record(record: dict[str, Any], root: Path, where: str) -> None:
    _keys(record, RECORD_KEYS, where)
    if record["schema_version"] != 1 or record["record_type"] != "pr_evidence":
        raise EvidenceError(f"{where} has an unsupported schema")
    if type(record["github_pr"]) is not int or record["github_pr"] < 1:
        raise EvidenceError(f"{where}.github_pr is invalid")
    _ident(record["architectural_unit"], f"{where}.architectural_unit")
    identity = record["identity"]
    _keys(identity, {"base_commit", "implementation_commit", "implementation_tree", "merge_commit"}, f"{where}.identity")
    base = _git_oid(identity["base_commit"], f"{where}.base")
    head = _git_oid(identity["implementation_commit"], f"{where}.head")
    merge = _git_oid(identity["merge_commit"], f"{where}.merge")
    tree = _git_oid(identity["implementation_tree"], f"{where}.tree")
    if git(root, "rev-parse", f"{head}^{{tree}}") != tree or git(root, "rev-parse", f"{merge}^{{tree}}") != tree:
        raise EvidenceError(f"{where} tree binding is wrong")
    parents = str(git(root, "show", "-s", "--format=%P", merge)).split()
    if parents != [base, head]:
        raise EvidenceError(f"{where} merge parents are wrong")
    counts, _ = diff_facts(root, base, head)
    if record["classified_diff"] != counts:
        raise EvidenceError(f"{where} classified diff is wrong")
    artifacts = record["artifacts"]
    if type(artifacts) is not list:
        raise EvidenceError(f"{where}.artifacts must be a list")
    artifact_ids: set[str] = set()
    for index, item in enumerate(artifacts):
        _validate_artifact(item, root, head, f"{where}.artifacts[{index}]")
        if item["id"] in artifact_ids:
            raise EvidenceError(f"{where} repeats artifact {item['id']}")
        artifact_ids.add(item["id"])
    for field in ("claims", "exit_criteria"):
        if type(record[field]) is not list:
            raise EvidenceError(f"{where}.{field} must be a list")
        item_ids: set[str] = set()
        for index, item in enumerate(record[field]):
            _keys(item, {"artifact_ids", "id", "statement", "status"}, f"{where}.{field}[{index}]")
            _ident(item["id"], f"{where}.{field}[{index}].id")
            if item["status"] not in {"verified", "digest_only", "reported_unretained", "invalidated", "not_applicable"}:
                raise EvidenceError(f"{where}.{field}[{index}].status is invalid")
            if type(item["statement"]) is not str or not item["statement"]:
                raise EvidenceError(f"{where}.{field}[{index}].statement is empty")
            if type(item["artifact_ids"]) is not list or not set(item["artifact_ids"]) <= artifact_ids:
                raise EvidenceError(f"{where}.{field}[{index}] refers to an unknown artifact")
            if item["id"] in item_ids:
                raise EvidenceError(f"{where}.{field} repeats {item['id']}")
            item_ids.add(item["id"])
    if type(record["nonclaims"]) is not list or not record["nonclaims"]:
        raise EvidenceError(f"{where}.nonclaims must be a nonempty list")
    if len(record["nonclaims"]) != len(set(record["nonclaims"])):
        raise EvidenceError(f"{where}.nonclaims contains duplicates")
    for value in record["nonclaims"]:
        _ident(value, f"{where}.nonclaims")
    reviews = record["reviews"]
    if type(reviews) is not list or {item.get("phase") for item in reviews} != {"implementation", "adversarial", "confirmation"}:
        raise EvidenceError(f"{where} must record all three review phases")
    for index, item in enumerate(reviews):
        _keys(item, {"artifact_id", "phase", "status", "summary"}, f"{where}.reviews[{index}]")
        if item["status"] not in {"reported_unretained", "not_reported", "invalidated"}:
            raise EvidenceError(f"{where}.reviews[{index}].status is invalid")
        if item["artifact_id"] is not None and item["artifact_id"] not in artifact_ids:
            raise EvidenceError(f"{where}.reviews[{index}] refers to an unknown artifact")
        if type(item["summary"]) is not str or not item["summary"]:
            raise EvidenceError(f"{where}.reviews[{index}].summary is empty")
    if record["github_pr"] == 43 and not any(
        item["id"] == "tests.full-exact-head" and item["status"] == "invalidated"
        for item in record["claims"]
    ):
        raise EvidenceError("PR 43 must preserve the invalid exact-head full-suite claim")


def validate_contract_document(contract: dict[str, Any], where: str = "contract") -> None:
    _keys(contract, CONTRACT_KEYS, where)
    if contract["schema_version"] != 1 or contract["record_type"] != "scope_contract":
        raise EvidenceError(f"{where} has an unsupported schema")
    _ident(contract["contract_id"], f"{where}.contract_id")
    _ident(contract["architectural_unit"], f"{where}.architectural_unit")
    _git_oid(contract["base_commit"], f"{where}.base_commit")
    budget = contract["budget"]
    _keys(budget, {"exemption_policy", "production_additions_max", "test_additions_max"}, f"{where}.budget")
    if budget != {"exemption_policy": "none", "production_additions_max": 3200, "test_additions_max": 2400}:
        raise EvidenceError(f"{where} budget or exemption policy changed")
    paths = contract["allowed_paths"]
    _keys(paths, {"control_exact", "control_prefixes", "production", "test"}, f"{where}.allowed_paths")
    expected_paths = {
        "control_exact": [".github/workflows/referee-evidence.yml", "scripts/check_referee_evidence.py", "tests/test_referee_evidence.py"],
        "control_prefixes": ["evidence/referee-hardening/"],
        "production": ["optima/eval/calibration.py", "optima/eval/evidence_store.py", "optima/eval/qualification.py", "optima/eval/reference_quality.py", "optima/eval/scoring.py"],
        "test": ["tests/test_calibration.py", "tests/test_evidence_store.py", "tests/test_qualification.py", "tests/test_reference_quality.py", "tests/test_scoring.py"],
    }
    if paths != expected_paths:
        raise EvidenceError(f"{where} allowed paths changed")
    for field, values in paths.items():
        if type(values) is not list or values != sorted(set(values)) or any(not isinstance(v, str) or v.startswith("/") or ".." in v for v in values):
            raise EvidenceError(f"{where}.allowed_paths.{field} must be sorted unique relative paths")
    required = contract["required_in_place"]
    if required != [{"change": "modify", "path": "optima/eval/scoring.py"}]:
        raise EvidenceError(f"{where} must require in-place scoring replacement")


def _scope_kind(path: str, contract: dict[str, Any]) -> str | None:
    allowed = contract["allowed_paths"]
    if path in allowed["control_exact"] or any(path.startswith(prefix) for prefix in allowed["control_prefixes"]):
        return "control"
    if path in allowed["production"]:
        return "production"
    if path in allowed["test"]:
        return "test"
    return None


def validate_scope(contract: dict[str, Any], changes: dict[str, tuple[str, int, int]]) -> None:
    totals = {"production": 0, "test": 0}
    for path, (status, added, _deleted) in changes.items():
        kind = _scope_kind(path, contract)
        if kind is None:
            raise EvidenceError(f"path is outside frozen PR4a scope: {path}")
        if kind in totals:
            totals[kind] += added
    if totals["production"] > 3200 or totals["test"] > 2400:
        raise EvidenceError(f"PR4a line budget exceeded: {totals}")
    for item in contract["required_in_place"]:
        if changes.get(item["path"], (None, 0, 0))[0] != "M":
            raise EvidenceError(f"required in-place replacement is absent: {item['path']}")


def validate_repository(root: Path, *, records_only: bool = False, pr_base: str | None = None) -> None:
    evidence = root / "evidence/referee-hardening"
    load_json(evidence / "schema-v1.json")
    expected = set(range(40, 47))
    seen: set[int] = set()
    for path in sorted((evidence / "records").glob("pr-*.json")):
        record = load_json(path)
        validate_record(record, root, str(path.relative_to(root)))
        seen.add(record["github_pr"])
    if seen != expected:
        raise EvidenceError(f"historical evidence set differs: {sorted(seen ^ expected)}")
    contract_path = evidence / "contracts/pr4a.json"
    contract = load_json(contract_path)
    validate_contract_document(contract)
    if records_only:
        return
    relative_contract = str(contract_path.relative_to(root))
    if pr_base is not None:
        _git_oid(pr_base, "pull-request base")
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{pr_base}:{relative_contract}"],
            cwd=root, capture_output=True, check=False,
        ).returncode == 0
        if exists:
            if bytes(git(root, "show", f"{pr_base}:{relative_contract}", text=False)) != contract_path.read_bytes():
                raise EvidenceError("a closed scope contract cannot change")
            return
        if pr_base != contract["base_commit"]:
            raise EvidenceError("PR4a must retain its frozen base commit")
    head = str(git(root, "rev-parse", "HEAD"))
    base = contract["base_commit"]
    commits = str(git(root, "rev-list", "--reverse", f"{base}..{head}", "--", relative_contract)).splitlines()
    if not commits:
        raise EvidenceError("PR4a scope contract is not committed")
    freeze = commits[0]
    frozen_bytes = bytes(git(root, "show", f"{freeze}:{relative_contract}", text=False))
    if frozen_bytes != contract_path.read_bytes():
        raise EvidenceError("post-freeze contract modification is forbidden")
    _, freeze_status = diff_facts(root, base, freeze)
    if any(_scope_kind(path, contract) != "control" for path in freeze_status):
        raise EvidenceError("the PR4a contract was frozen after implementation began")
    counts, statuses = diff_facts(root, base, head)
    changes = {
        path: (status, counts[classify(path)]["added"], counts[classify(path)]["deleted"])
        for path, status in statuses.items()
    }
    # Use per-path numstat rather than category totals for the scope budget.
    for line in str(git(root, "diff", "--numstat", "--no-renames", f"{base}..{head}")).splitlines():
        added, deleted, path = line.split("\t", 2)
        changes[path] = (statuses[path], int(added), int(deleted))
    validate_scope(contract, changes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--records-only", action="store_true")
    parser.add_argument("--pr-base")
    args = parser.parse_args()
    try:
        validate_repository(args.root.resolve(), records_only=args.records_only, pr_base=args.pr_base)
    except EvidenceError as exc:
        parser.error(str(exc))
    print("referee evidence: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
