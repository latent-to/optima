#!/usr/bin/env python3
"""Validate referee evidence records and the frozen change contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
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
CONTRACT_KEYS_V1 = {
    "allowed_paths", "architectural_unit", "base_commit", "budget", "contract_id",
    "record_type", "required_in_place", "schema_version",
}
CONTRACT_KEYS_V2 = CONTRACT_KEYS_V1 | {"authority_boundary"}

PR4B_BASE = "9a34b68a58ff53a0c04273a21f86de9e9467db80"
PR4B_PRODUCTION = [
    "optima/eval/_launch.py",
    "optima/eval/calibration.py",
    "optima/eval/engine_worker.py",
    "optima/eval/oci_backend.py",
    "optima/eval/oci_process.py",
    "optima/eval/oci_reference_session.py",
    "optima/eval/oci_session_worker.py",
    "optima/eval/qualification.py",
    "optima/eval/reference_protocol.py",
    "optima/eval/reference_quality.py",
]
PR4B_TESTS = [
    "tests/test_calibration.py",
    "tests/test_isolation.py",
    "tests/test_launch_execution_receipts.py",
    "tests/test_oci_backend.py",
    "tests/test_oci_process.py",
    "tests/test_oci_reference_session.py",
    "tests/test_oci_session_worker_order.py",
    "tests/test_qualification.py",
    "tests/test_reference_protocol.py",
    "tests/test_reference_quality.py",
]
PR4B_AUTHORITY_ROOTS = [
    "optima.eval.calibration",
    "optima.eval.marginal_runtime",
    "optima.eval.oci_backend",
    "optima.eval.oci_process",
    "optima.eval.oci_reference_session",
    "optima.eval.qualification",
    "optima.eval.reference_protocol",
    "optima.eval.reference_quality",
    "optima.eval.scoring",
]
PR4B_FORBIDDEN_MODULES = [
    "optima.audit",
    "optima.eval._launch",
    "optima.eval.capability",
    "optima.eval.throughput_kl",
]
PR4C_BASE = "f4a68d1a7ecf9c21f3ee6e765e1ad596f108764e"
PR4C_PRODUCTION = [
    "optima/eval/oci_backend.py",
    "optima/eval/oci_process.py",
    "optima/eval/qualification.py",
    "optima/eval/qualification_runner.py",
    "optima/eval/reference_quality.py",
]
PR4C_TESTS = [
    "tests/test_oci_backend.py",
    "tests/test_oci_process.py",
    "tests/test_qualification.py",
    "tests/test_qualification_runner.py",
    "tests/test_reference_quality.py",
]
PR4C_AUTHORITY_ROOTS = ["optima.eval.qualification_runner"]
PR4C_FORBIDDEN_MODULES = [
    "optima.audit",
    "optima.chain.validator_loop",
    "optima.cli",
    "optima.eval._launch",
    "optima.eval.capability",
    "optima.eval.throughput_kl",
]
PR4D_BASE = "7b45146a9c7c4e21859fe2878fc3e62a644725e4"
PR4D_PRODUCTION = [
    "optima/eval/engine_worker.py",
    "optima/eval/oci_session_protocol.py",
    "optima/eval/oci_session_worker.py",
    "optima/seams.py",
]
PR4D_TESTS = [
    "tests/test_oci_session_protocol.py",
    "tests/test_oci_session_worker_order.py",
]
PR4D_AUTHORITY_ROOTS = ["optima.eval.oci_session_protocol"]
PR4D_FORBIDDEN_MODULES = PR4C_FORBIDDEN_MODULES


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


def _validate_artifact(
    item: dict[str, Any], root: Path, commit: str, schema_version: int, where: str
) -> None:
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
    if schema_version == 1:
        raw = bytes(git(root, "show", f"{commit}:{item['locator']}", text=False))
    else:
        locator = item["locator"]
        if not locator.startswith("evidence/referee-hardening/artifacts/"):
            raise EvidenceError(f"{where} v2 repository artifact must use the evidence artifact directory")
        candidate = root / locator
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root.resolve())
        except (OSError, ValueError) as exc:
            raise EvidenceError(f"{where} repository artifact escapes or is absent") from exc
        if candidate.is_symlink() or not candidate.is_file():
            raise EvidenceError(f"{where} repository artifact must be a regular file")
        raw = candidate.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest or len(raw) != item["size_bytes"]:
        raise EvidenceError(f"{where} repository artifact bytes do not match")
    if item["reason"] is not None:
        raise EvidenceError(f"{where} retained artifact cannot have an absence reason")


def validate_record(record: dict[str, Any], root: Path, where: str) -> None:
    _keys(record, RECORD_KEYS, where)
    schema_version = record["schema_version"]
    if schema_version not in {1, 2} or record["record_type"] != "pr_evidence":
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
        _validate_artifact(item, root, head, schema_version, f"{where}.artifacts[{index}]")
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
    if record["github_pr"] in {47, 48, 49}:
        if schema_version != 2:
            raise EvidenceError("retained referee PR evidence must use schema v2")
        retained = {
            item["id"] for item in artifacts
            if item["availability"] == "repository"
        }
        if "github.referee-evidence" not in retained:
            raise EvidenceError("referee PR must retain the GitHub evidence check receipt")


def validate_contract_document(contract: dict[str, Any], where: str = "contract") -> None:
    schema_version = contract.get("schema_version")
    if schema_version not in {1, 2}:
        raise EvidenceError(f"{where} has an unsupported schema")
    expected_keys = CONTRACT_KEYS_V1 if schema_version == 1 else CONTRACT_KEYS_V2
    _keys(contract, expected_keys, where)
    if contract["record_type"] != "scope_contract":
        raise EvidenceError(f"{where} has an unsupported schema")
    _ident(contract["contract_id"], f"{where}.contract_id")
    _ident(contract["architectural_unit"], f"{where}.architectural_unit")
    _git_oid(contract["base_commit"], f"{where}.base_commit")
    control_exact = [
        ".github/workflows/referee-evidence.yml",
        "scripts/check_referee_evidence.py",
        "tests/test_referee_evidence.py",
    ]
    specs = {
        "pr4a": {
            "schema_version": 1,
            "architectural_unit": "4a",
            "base_commit": "17ecdeb5213d03771964939d80da9343618a7e86",
            "budget": {"exemption_policy": "none", "production_additions_max": 3200, "test_additions_max": 2400},
            "production": ["optima/eval/calibration.py", "optima/eval/evidence_store.py", "optima/eval/qualification.py", "optima/eval/reference_quality.py", "optima/eval/scoring.py"],
            "test": ["tests/test_calibration.py", "tests/test_evidence_store.py", "tests/test_qualification.py", "tests/test_reference_quality.py", "tests/test_scoring.py"],
            "required": [{"change": "modify", "path": "optima/eval/scoring.py"}],
            "authority": None,
        },
        "pr4b": {
            "schema_version": 2,
            "architectural_unit": "4b",
            "base_commit": PR4B_BASE,
            "budget": {"exemption_policy": "none", "production_additions_max": 3300, "test_additions_max": 2600},
            "production": PR4B_PRODUCTION,
            "test": PR4B_TESTS,
            "required": [
                {"change": "modify", "path": "optima/eval/_launch.py"},
                {"change": "modify", "path": "optima/eval/calibration.py"},
                {"change": "add", "path": "optima/eval/engine_worker.py"},
                {"change": "modify", "path": "optima/eval/oci_backend.py"},
                {"change": "modify", "path": "optima/eval/oci_process.py"},
                {"change": "add", "path": "optima/eval/oci_reference_session.py"},
                {"change": "modify", "path": "optima/eval/oci_session_worker.py"},
                {"change": "modify", "path": "optima/eval/qualification.py"},
                {"change": "add", "path": "optima/eval/reference_protocol.py"},
                {"change": "modify", "path": "optima/eval/reference_quality.py"},
            ],
            "authority": {"forbidden_modules": PR4B_FORBIDDEN_MODULES, "roots": PR4B_AUTHORITY_ROOTS},
        },
        "pr4c": {
            "schema_version": 2,
            "architectural_unit": "4c",
            "base_commit": PR4C_BASE,
            "budget": {"exemption_policy": "none", "production_additions_max": 1400, "test_additions_max": 1800},
            "production": PR4C_PRODUCTION,
            "test": PR4C_TESTS,
            "required": [
                {"change": "modify", "path": "optima/eval/oci_backend.py"},
                {"change": "modify", "path": "optima/eval/oci_process.py"},
                {"change": "modify", "path": "optima/eval/qualification.py"},
                {"change": "add", "path": "optima/eval/qualification_runner.py"},
                {"change": "modify", "path": "optima/eval/reference_quality.py"},
            ],
            "authority": {"forbidden_modules": PR4C_FORBIDDEN_MODULES, "roots": PR4C_AUTHORITY_ROOTS},
        },
        "pr4d": {
            "schema_version": 2,
            "architectural_unit": "4d",
            "base_commit": PR4D_BASE,
            "budget": {"exemption_policy": "none", "production_additions_max": 250, "test_additions_max": 250},
            "production": PR4D_PRODUCTION,
            "test": PR4D_TESTS,
            "required": [
                {"change": "modify", "path": "optima/eval/engine_worker.py"},
                {"change": "modify", "path": "optima/eval/oci_session_protocol.py"},
                {"change": "modify", "path": "optima/eval/oci_session_worker.py"},
                {"change": "modify", "path": "optima/seams.py"},
            ],
            "authority": {"forbidden_modules": PR4D_FORBIDDEN_MODULES, "roots": PR4D_AUTHORITY_ROOTS},
        },
    }
    spec = specs.get(contract["contract_id"])
    if spec is None or schema_version != spec["schema_version"]:
        raise EvidenceError(f"{where}.contract_id or schema changed")
    expected_identity = {
        "architectural_unit": spec["architectural_unit"],
        "base_commit": spec["base_commit"],
        "contract_id": contract["contract_id"],
    }
    for field, expected in expected_identity.items():
        if contract[field] != expected:
            raise EvidenceError(f"{where}.{field} changed")
    budget = contract["budget"]
    _keys(budget, {"exemption_policy", "production_additions_max", "test_additions_max"}, f"{where}.budget")
    if budget != spec["budget"]:
        raise EvidenceError(f"{where} budget or exemption policy changed")
    paths = contract["allowed_paths"]
    _keys(paths, {"control_exact", "control_prefixes", "production", "test"}, f"{where}.allowed_paths")
    expected_paths = {
        "control_exact": control_exact,
        "control_prefixes": ["evidence/referee-hardening/"],
        "production": spec["production"],
        "test": spec["test"],
    }
    if paths != expected_paths:
        raise EvidenceError(f"{where} allowed paths changed")
    for field, values in paths.items():
        if type(values) is not list or values != sorted(set(values)) or any(not isinstance(v, str) or v.startswith("/") or ".." in v for v in values):
            raise EvidenceError(f"{where}.allowed_paths.{field} must be sorted unique relative paths")
    required = contract["required_in_place"]
    if required != spec["required"]:
        raise EvidenceError(f"{where} required replacements changed")
    if schema_version == 2:
        authority = contract["authority_boundary"]
        _keys(authority, {"forbidden_modules", "roots"}, f"{where}.authority_boundary")
        if authority != spec["authority"]:
            raise EvidenceError(f"{where} authority boundary changed")


def validate_v2_schema_contract(schema: dict[str, Any], contract: dict[str, Any]) -> None:
    """Require one generic v2 scope branch; exact values live in frozen contracts."""

    branches = schema.get("oneOf")
    if not isinstance(branches, list):
        raise EvidenceError("schema-v2 oneOf is absent")
    matches = []
    for branch in branches:
        properties = branch.get("properties", {}) if isinstance(branch, dict) else {}
        record = properties.get("record_type", {})
        version = properties.get("schema_version", {})
        if record.get("const") == "scope_contract" and version.get("const") == 2:
            matches.append(properties)
    if len(matches) != 1:
        raise EvidenceError("schema-v2 scope-contract branch is absent or ambiguous")
    properties = matches[0]
    if set(properties) != CONTRACT_KEYS_V2 or any(
        field not in properties for field in contract
    ):
        raise EvidenceError("schema-v2 scope fields differ")
    for field in CONTRACT_KEYS_V2 - {"record_type", "schema_version"}:
        if "const" in properties[field]:
            raise EvidenceError(f"schema-v2 scope field must be generic: {field}")


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
            raise EvidenceError(f"path is outside frozen {contract['contract_id']} scope: {path}")
        if kind in totals:
            totals[kind] += added
    budget = contract["budget"]
    if (
        totals["production"] > budget["production_additions_max"]
        or totals["test"] > budget["test_additions_max"]
    ):
        raise EvidenceError(f"{contract['contract_id']} line budget exceeded: {totals}")
    for item in contract["required_in_place"]:
        expected_status = {"add": "A", "modify": "M"}[item["change"]]
        if changes.get(item["path"], (None, 0, 0))[0] != expected_status:
            raise EvidenceError(f"required {item['change']} is absent: {item['path']}")


def _repo_modules(root: Path) -> dict[str, tuple[Path, bool]]:
    package_root = root / "optima"
    modules: dict[str, tuple[Path, bool]] = {}
    if not package_root.is_dir():
        return modules
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(root).with_suffix("")
        parts = list(relative.parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
        if parts:
            modules[".".join(parts)] = (path, is_package)
    return modules


def _local_imports(
    module: str,
    path: Path,
    is_package: bool,
    modules: dict[str, tuple[Path, bool]],
) -> set[str]:
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise EvidenceError(f"cannot parse authority module {module}: {exc}") from exc
    package = module if is_package else module.rpartition(".")[0]
    imported: set[str] = set()
    import_module_names = {"import_module"}
    builtin_import_names = {"__import__"}
    importlib_names = {"importlib"}
    builtins_names = {"builtins"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            importlib_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "importlib"
            )
            builtins_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "builtins"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            import_module_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "import_module"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "builtins":
            builtin_import_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "__import__"
            )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in modules:
                    imported.add(alias.name)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            if not package:
                raise EvidenceError(f"relative import has no package in {module}:{node.lineno}")
            relative = "." * node.level + (node.module or "")
            try:
                base = importlib.util.resolve_name(relative, package)
            except (ImportError, ValueError) as exc:
                raise EvidenceError(f"invalid relative import in {module}:{node.lineno}") from exc
        else:
            base = node.module or ""
        if base in modules:
            imported.add(base)
        for alias in node.names:
            if alias.name == "*":
                continue
            candidate = f"{base}.{alias.name}" if base else alias.name
            if candidate in modules:
                imported.add(candidate)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        import_module_call = (
            isinstance(node.func, ast.Name) and node.func.id in import_module_names
        ) or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in importlib_names
        )
        builtin_import_call = (
            isinstance(node.func, ast.Name) and node.func.id in builtin_import_names
        ) or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "__import__"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in builtins_names
        )
        if not (import_module_call or builtin_import_call):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(
            node.args[0].value, str
        ):
            raise EvidenceError(
                f"authority module has an unresolved dynamic import in {module}:{node.lineno}"
            )
        dynamic = node.args[0].value
        if import_module_call:
            if len(node.args) > 2 or any(
                keyword.arg not in {"package"} for keyword in node.keywords
            ):
                raise EvidenceError(
                    f"authority module has an unresolved dynamic import in {module}:{node.lineno}"
                )
            package_nodes = [
                *node.args[1:2],
                *(keyword.value for keyword in node.keywords if keyword.arg == "package"),
            ]
            if len(package_nodes) > 1:
                raise EvidenceError(
                    f"authority module has an unresolved dynamic import in {module}:{node.lineno}"
                )
            if dynamic.startswith("."):
                if not package_nodes:
                    raise EvidenceError(
                        f"authority module has an unresolved dynamic import in {module}:{node.lineno}"
                    )
                package_node = package_nodes[0]
                if isinstance(package_node, ast.Name) and package_node.id == "__package__":
                    dynamic_package = package
                elif isinstance(package_node, ast.Constant) and isinstance(
                    package_node.value, str
                ):
                    dynamic_package = package_node.value
                else:
                    raise EvidenceError(
                        f"authority module has an unresolved dynamic import in {module}:{node.lineno}"
                    )
                try:
                    dynamic = importlib.util.resolve_name(dynamic, dynamic_package)
                except (ImportError, ValueError) as exc:
                    raise EvidenceError(
                        f"invalid dynamic relative import in {module}:{node.lineno}"
                    ) from exc
            elif package_nodes and not (
                isinstance(package_nodes[0], ast.Name)
                and package_nodes[0].id == "__package__"
            ) and not (
                isinstance(package_nodes[0], ast.Constant)
                and isinstance(package_nodes[0].value, str)
            ):
                raise EvidenceError(
                    f"authority module has an unresolved dynamic import in {module}:{node.lineno}"
                )
            candidates = (dynamic,)
        else:
            if len(node.args) > 5 or any(
                keyword.arg not in {"globals", "locals", "fromlist", "level"}
                for keyword in node.keywords
            ):
                raise EvidenceError(
                    f"authority module has an unresolved dynamic import in {module}:{node.lineno}"
                )
            keyword_values = {keyword.arg: keyword.value for keyword in node.keywords}
            level_node = keyword_values.get(
                "level", node.args[4] if len(node.args) > 4 else ast.Constant(value=0)
            )
            if not isinstance(level_node, ast.Constant) or type(level_node.value) is not int:
                raise EvidenceError(
                    f"authority module has an unresolved dynamic import in {module}:{node.lineno}"
                )
            if level_node.value:
                raise EvidenceError(
                    f"authority module has an unresolved dynamic import in {module}:{node.lineno}"
                )
            fromlist_node = keyword_values.get(
                "fromlist", node.args[3] if len(node.args) > 3 else ast.Tuple(elts=[])
            )
            if not isinstance(fromlist_node, (ast.Tuple, ast.List)) or any(
                not isinstance(item, ast.Constant) or not isinstance(item.value, str)
                for item in fromlist_node.elts
            ):
                raise EvidenceError(
                    f"authority module has an unresolved dynamic import in {module}:{node.lineno}"
                )
            candidates = (dynamic, *(
                f"{dynamic}.{item.value}" for item in fromlist_node.elts if item.value != "*"
            ))
        imported.update(candidate for candidate in candidates if candidate in modules)
    return imported


def validate_authority_boundary(root: Path, contract: dict[str, Any]) -> None:
    authority = contract.get("authority_boundary")
    if authority is None:
        return
    modules = _repo_modules(root)
    forbidden = set(authority["forbidden_modules"])
    cache: dict[str, set[str]] = {}
    for authority_root in authority["roots"]:
        if authority_root not in modules:
            raise EvidenceError(f"authority root is absent: {authority_root}")
        stack: list[tuple[str, tuple[str, ...]]] = [(authority_root, (authority_root,))]
        visited: set[str] = set()
        while stack:
            module, chain = stack.pop()
            if module in visited:
                continue
            visited.add(module)
            if module in forbidden:
                raise EvidenceError(
                    f"authority root {authority_root} reaches forbidden module {module}: "
                    + " -> ".join(chain)
                )
            if module not in cache:
                path, is_package = modules[module]
                cache[module] = _local_imports(module, path, is_package, modules)
            for dependency in sorted(cache[module], reverse=True):
                stack.append((dependency, (*chain, dependency)))


def _git_path_exists(root: Path, commit: str, relative_path: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:{relative_path}"],
        cwd=root,
        capture_output=True,
        check=False,
    ).returncode == 0


def _require_frozen_bytes(root: Path, commit: str, relative_path: str, label: str) -> None:
    path = root / relative_path
    try:
        current = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"{label} is absent: {relative_path}") from exc
    if bytes(git(root, "show", f"{commit}:{relative_path}", text=False)) != current:
        raise EvidenceError(f"{label} changed after freeze: {relative_path}")


def validate_repository(root: Path, *, records_only: bool = False, pr_base: str | None = None) -> None:
    evidence = root / "evidence/referee-hardening"
    load_json(evidence / "schema-v1.json")
    schema_v2 = load_json(evidence / "schema-v2.json")
    expected = set(range(40, 50))
    seen: set[int] = set()
    for path in sorted((evidence / "records").glob("pr-*.json")):
        record = load_json(path)
        validate_record(record, root, str(path.relative_to(root)))
        seen.add(record["github_pr"])
    if seen != expected:
        raise EvidenceError(f"historical evidence set differs: {sorted(seen ^ expected)}")
    contracts: list[tuple[Path, dict[str, Any]]] = []
    contract_ids: set[str] = set()
    for contract_path in sorted((evidence / "contracts").glob("*.json")):
        contract = load_json(contract_path)
        validate_contract_document(contract, str(contract_path.relative_to(root)))
        if contract["schema_version"] == 2:
            validate_v2_schema_contract(schema_v2, contract)
        if contract["contract_id"] in contract_ids:
            raise EvidenceError(f"duplicate scope contract: {contract['contract_id']}")
        contract_ids.add(contract["contract_id"])
        contracts.append((contract_path, contract))
    if contract_ids != {"pr4a", "pr4b", "pr4c", "pr4d"}:
        raise EvidenceError(f"scope contract set differs: {sorted(contract_ids)}")
    if records_only:
        return
    if pr_base is not None:
        _git_oid(pr_base, "pull-request base")

    active: list[tuple[Path, dict[str, Any]]] = []
    witness_commits = [contract["base_commit"] for _, contract in contracts]
    if pr_base is not None:
        witness_commits.insert(0, pr_base)
    for contract_path, contract in contracts:
        relative_contract = str(contract_path.relative_to(root))
        witnesses = [
            commit for commit in witness_commits
            if commit != contract["base_commit"] and _git_path_exists(root, commit, relative_contract)
        ]
        if witnesses:
            for witness in witnesses:
                _require_frozen_bytes(root, witness, relative_contract, "closed scope contract")
            continue
        if pr_base is not None and contract["base_commit"] != pr_base:
            raise EvidenceError(
                f"{contract['contract_id']} is neither closed nor based on the pull-request base"
            )
        active.append((contract_path, contract))
    if not active:
        return
    if len(active) != 1:
        raise EvidenceError(f"exactly one scope contract must be active: {[item[1]['contract_id'] for item in active]}")
    contract_path, contract = active[0]
    relative_contract = str(contract_path.relative_to(root))
    head = str(git(root, "rev-parse", "HEAD"))
    base = contract["base_commit"]
    commits = str(git(root, "rev-list", "--reverse", f"{base}..{head}", "--", relative_contract)).splitlines()
    if not commits:
        raise EvidenceError(f"{contract['contract_id']} scope contract is not committed")
    freeze = commits[0]
    frozen_bytes = bytes(git(root, "show", f"{freeze}:{relative_contract}", text=False))
    if frozen_bytes != contract_path.read_bytes():
        raise EvidenceError("post-freeze contract modification is forbidden")
    _, freeze_status = diff_facts(root, base, freeze)
    if any(_scope_kind(path, contract) != "control" for path in freeze_status):
        raise EvidenceError(f"the {contract['contract_id']} contract was frozen after implementation began")
    if contract["schema_version"] == 2:
        for path in freeze_status:
            _require_frozen_bytes(root, freeze, path, "referee control file")
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
    validate_authority_boundary(root, contract)


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
