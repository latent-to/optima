#!/usr/bin/env python3
"""Validate repository-local documentation contracts.

MkDocs' strict build validates rendering. This checker covers repository
invariants that MkDocs cannot infer: navigation ownership, source-link
existence, CLI inventory freshness, and the absence of machine-private or
retired-repository references.
"""

from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import markdown
import yaml


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MKDOCS_CONFIG = ROOT / "mkdocs.yml"
CLI_SOURCE = ROOT / "optima" / "cli.py"
CLI_REFERENCE = DOCS / "reference" / "cli.md"
# This gitignored local operations log can contain private infrastructure
# context. It is deliberately outside both validation and publication.
PRIVATE_DOCS = {DOCS / "WORKLOG.md"}
REDIRECT_MARKER = re.compile(r"\A<!-- docs-redirect: ([^>\n]+) -->\n")

PRIVATE_PATTERNS = (
    (re.compile(r"/Users/[A-Za-z0-9._-]+/"), "absolute macOS home path"),
    (re.compile(r"/home/[A-Za-z0-9._-]+/"), "absolute Linux home path"),
    (re.compile(r"/root/(?:[A-Za-z0-9._-]+/)?"), "absolute root home path"),
    (
        re.compile(r"""(?:\]\(|href=["'])file://""", re.IGNORECASE),
        "clickable local file URL",
    ),
    (
        re.compile(r"(?:^|\s)ssh\s+(?:root|ubuntu)@", re.MULTILINE),
        "direct privileged SSH endpoint",
    ),
    (
        re.compile(r"(?:~/(?:\.claude/projects|\.codex/sessions)|AgentArchive/)"),
        "private agent-session path",
    ),
)

RETIRED_REPOSITORY_PATTERNS = (
    re.compile(
        r"https?://github\.com/latent-to/(?:optima-docs|optima)(?:[/.#?]|$)"
    ),
    re.compile(r"git@github\.com:latent-to/(?:optima-docs|optima)(?:\.git)?\b"),
    re.compile(r"https?://latent-to\.github\.io/optima-docs(?:[/#?]|$)"),
)

CLI_TABLE_ROW = re.compile(r"^\|\s*`([a-z][a-z0-9-]*)`\s*\|", re.MULTILINE)


class _RenderedLinkParser(HTMLParser):
    """Collect links and images from Python-Markdown's rendered output."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.targets: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        for name, value in attrs:
            if name == "id" and value:
                self.ids.add(value)
        attribute = "href" if tag == "a" else "src" if tag == "img" else None
        if attribute is None:
            return
        for name, value in attrs:
            if name == attribute and value:
                self.targets.append(value)


def markdown_paths() -> list[Path]:
    """Return every Markdown source that participates in repository docs."""

    paths = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    paths.extend(
        path for path in sorted(DOCS.rglob("*.md")) if path not in PRIVATE_DOCS
    )
    return [path for path in paths if path.is_file()]


def _markdown_parser() -> markdown.Markdown:
    # These extensions are sufficient to reproduce heading IDs and link output.
    # The strict MkDocs build remains the authority for the full Material stack.
    return markdown.Markdown(
        extensions=[
            "abbr",
            "admonition",
            "attr_list",
            "def_list",
            "fenced_code",
            "footnotes",
            "md_in_html",
            "tables",
            "toc",
        ]
    )


def rendered_links(path: Path) -> list[str]:
    parser = _RenderedLinkParser()
    rendered = _markdown_parser().convert(path.read_text(encoding="utf-8"))
    parser.feed(rendered)
    return parser.targets


def rendered_ids(path: Path) -> set[str]:
    parser = _RenderedLinkParser()
    rendered = _markdown_parser().convert(path.read_text(encoding="utf-8"))
    parser.feed(rendered)
    return parser.ids


def _repository_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_redirect_pages(failures: list[str]) -> set[Path]:
    """Validate root-level compatibility pages and return their source paths."""

    redirects: set[Path] = set()
    for path in sorted(DOCS.rglob("*.md")):
        if path in PRIVATE_DOCS:
            continue
        text = path.read_text(encoding="utf-8")
        match = REDIRECT_MARKER.match(text)
        if match is None:
            if "docs-redirect:" in text.split("\n", 1)[0]:
                failures.append(
                    f"{_repository_relative(path)}: malformed docs-redirect marker"
                )
            continue
        if path.parent != DOCS:
            failures.append(
                f"{_repository_relative(path)}: compatibility pages must be "
                "root-level docs"
            )
            continue

        raw_target = match.group(1)
        target = Path(raw_target)
        if (
            target.is_absolute()
            or target.suffix.lower() != ".md"
            or ".." in target.parts
        ):
            failures.append(
                f"{_repository_relative(path)}: invalid redirect target: {raw_target}"
            )
            continue

        candidate = (DOCS / target).resolve()
        try:
            candidate.relative_to(DOCS)
        except ValueError:
            failures.append(
                f"{_repository_relative(path)}: redirect escapes docs/: {raw_target}"
            )
            continue
        if not candidate.is_file():
            failures.append(
                f"{_repository_relative(path)}: missing redirect target: {raw_target}"
            )
            continue
        if candidate == path or candidate in PRIVATE_DOCS:
            failures.append(
                f"{_repository_relative(path)}: redirect target is not a public "
                f"canonical page: {raw_target}"
            )
            continue
        if REDIRECT_MARKER.match(candidate.read_text(encoding="utf-8")):
            failures.append(
                f"{_repository_relative(path)}: redirect chaining is not allowed"
            )
            continue

        expected = (
            f"<!-- docs-redirect: {raw_target} -->\n\n"
            "# Documentation moved\n\n"
            "This compatibility path now points to "
            f"[canonical documentation]({raw_target}).\n"
            "It contains no independent guidance.\n"
        )
        if text != expected:
            failures.append(
                f"{_repository_relative(path)}: compatibility page contains "
                "content beyond the canonical redirect template"
            )
            continue
        redirects.add(path)
    return redirects


def check_private_and_retired_references(
    paths: list[Path], failures: list[str]
) -> None:
    text_paths = [*paths, MKDOCS_CONFIG]
    for path in text_paths:
        text = path.read_text(encoding="utf-8")
        relative = _repository_relative(path)
        for pattern, label in PRIVATE_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{relative}:{line}: {label}: {match.group(0).strip()}"
                )
        for pattern in RETIRED_REPOSITORY_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{relative}:{line}: retired repository URL: {match.group(0)}"
                )


def check_internal_links(paths: list[Path], failures: list[str]) -> int:
    anchors = {path.resolve(): rendered_ids(path) for path in paths}
    checked = 0
    for path in paths:
        relative = _repository_relative(path)
        for target in rendered_links(path):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith("//"):
                continue
            checked += 1
            linked_path = unquote(parsed.path)
            candidate = (
                (path.parent / linked_path).resolve()
                if linked_path
                else path.resolve()
            )
            if linked_path and not candidate.exists():
                route = Path(linked_path.rstrip("/"))
                route_candidates = [
                    (path.parent / route).with_suffix(".md").resolve(),
                    (path.parent / route / "index.md").resolve(),
                ]
                candidate = next(
                    (
                        route_candidate
                        for route_candidate in route_candidates
                        if route_candidate.exists()
                    ),
                    candidate,
                )
            try:
                candidate.relative_to(ROOT)
            except ValueError:
                failures.append(
                    f"{relative}: internal link escapes repository: {target}"
                )
                continue
            if path.is_relative_to(DOCS) and not candidate.is_relative_to(DOCS):
                failures.append(
                    f"{relative}: published-doc link escapes docs/: {target}; "
                    "use a canonical repository URL"
                )
                continue
            if not candidate.exists():
                failures.append(
                    f"{relative}: missing internal link target: {target}"
                )
                continue
            fragment = unquote(parsed.fragment)
            if (
                fragment
                and candidate.is_file()
                and candidate.suffix.lower() == ".md"
                and fragment not in anchors.get(candidate, set())
            ):
                failures.append(
                    f"{relative}: missing local heading: {target}"
                )
    return checked


def _flatten_nav(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_flatten_nav(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_flatten_nav(item))
        return result
    return []


def check_nav_coverage(redirects: set[Path], failures: list[str]) -> int:
    # BaseLoader intentionally treats the pymdown Python-name tag as a scalar.
    # Navigation needs only strings, lists, and mappings.
    config = yaml.load(
        MKDOCS_CONFIG.read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    if not isinstance(config, dict) or "nav" not in config:
        failures.append("mkdocs.yml: missing nav configuration")
        return 0

    expected_exclusions = {
        path.relative_to(DOCS).as_posix() for path in PRIVATE_DOCS
    }
    configured_exclusion_rows = [
        line.strip()
        for line in str(config.get("exclude_docs", "")).splitlines()
        if line.strip()
    ]
    duplicate_exclusions = sorted(
        path
        for path, count in Counter(configured_exclusion_rows).items()
        if count > 1
    )
    for path in duplicate_exclusions:
        failures.append(f"mkdocs.yml: duplicate exclude_docs entry: {path}")
    configured_exclusions = set(configured_exclusion_rows)
    if configured_exclusions != expected_exclusions:
        failures.append(
            "mkdocs.yml: exclude_docs must contain exactly the private local docs: "
            + ", ".join(sorted(expected_exclusions))
        )

    expected_not_in_nav = {
        path.relative_to(DOCS).as_posix() for path in redirects
    }
    configured_not_in_nav_rows = [
        line.strip()
        for line in str(config.get("not_in_nav", "")).splitlines()
        if line.strip()
    ]
    duplicate_not_in_nav = sorted(
        path
        for path, count in Counter(configured_not_in_nav_rows).items()
        if count > 1
    )
    for path in duplicate_not_in_nav:
        failures.append(f"mkdocs.yml: duplicate not_in_nav entry: {path}")
    configured_not_in_nav = set(configured_not_in_nav_rows)
    if configured_not_in_nav != expected_not_in_nav:
        failures.append(
            "mkdocs.yml: not_in_nav must contain exactly the validated "
            "compatibility pages: "
            + ", ".join(sorted(expected_not_in_nav))
        )

    entries: list[str] = []
    for target in _flatten_nav(config["nav"]):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or target.startswith("//"):
            continue
        nav_path = unquote(parsed.path)
        if not nav_path:
            failures.append(f"mkdocs.yml: empty local nav target: {target!r}")
            continue
        candidate = (DOCS / nav_path).resolve()
        try:
            relative = candidate.relative_to(DOCS).as_posix()
        except ValueError:
            failures.append(f"mkdocs.yml: nav target escapes docs/: {target}")
            continue
        if not candidate.is_file():
            failures.append(f"mkdocs.yml: missing nav target: {target}")
            continue
        if candidate.suffix.lower() != ".md":
            failures.append(f"mkdocs.yml: local nav target is not Markdown: {target}")
            continue
        entries.append(relative)

    duplicates = sorted(
        path for path, count in Counter(entries).items() if count > 1
    )
    for path in duplicates:
        failures.append(f"mkdocs.yml: duplicate nav target: {path}")

    redirect_entries = {
        path.relative_to(DOCS).as_posix() for path in redirects
    }
    for path in sorted(set(entries) & redirect_entries):
        failures.append(
            f"mkdocs.yml: compatibility page must not appear in nav: {path}"
        )

    discovered = {
        path.relative_to(DOCS).as_posix()
        for path in DOCS.rglob("*.md")
        if path not in PRIVATE_DOCS and path not in redirects
    }
    orphans = sorted(discovered - set(entries))
    for path in orphans:
        failures.append(f"docs/{path}: page is not represented in mkdocs nav")
    return len(entries)


def _cli_source_commands() -> set[str]:
    tree = ast.parse(CLI_SOURCE.read_text(encoding="utf-8"), filename=str(CLI_SOURCE))
    commands: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        if not isinstance(function, ast.Attribute) or function.attr != "add_parser":
            continue
        owner = function.value
        command = node.args[0]
        if (
            isinstance(owner, ast.Name)
            and owner.id == "sub"
            and isinstance(command, ast.Constant)
            and isinstance(command.value, str)
        ):
            commands.add(command.value)
    return commands


def check_cli_inventory(failures: list[str]) -> int:
    source_commands = _cli_source_commands()
    if not CLI_REFERENCE.is_file():
        failures.append("docs/reference/cli.md: missing CLI reference")
        return len(source_commands)

    text = CLI_REFERENCE.read_text(encoding="utf-8")
    inventory_start = text.find("## Command inventory")
    if inventory_start < 0:
        failures.append("docs/reference/cli.md: missing 'Command inventory' section")
        return len(source_commands)
    inventory = text[inventory_start:]
    next_section = inventory.find("\n## ", 1)
    if next_section >= 0:
        inventory = inventory[:next_section]
    documented_rows = CLI_TABLE_ROW.findall(inventory)
    documented_commands = set(documented_rows)
    duplicates = sorted(
        command
        for command, count in Counter(documented_rows).items()
        if count > 1
    )
    if duplicates:
        failures.append(
            "docs/reference/cli.md: duplicate commands in inventory: "
            + ", ".join(duplicates)
        )

    missing = sorted(source_commands - documented_commands)
    stale = sorted(documented_commands - source_commands)
    if missing:
        failures.append(
            "docs/reference/cli.md: commands missing from inventory: "
            + ", ".join(missing)
        )
    if stale:
        failures.append(
            "docs/reference/cli.md: retired commands in inventory: "
            + ", ".join(stale)
        )
    return len(source_commands)


def check_cacheon_main_links(paths: list[Path], failures: list[str]) -> int:
    checked = 0
    for path in paths:
        relative = _repository_relative(path)
        for target in rendered_links(path):
            parsed = urlsplit(target)
            prefix = "/latent-to/cacheon/"
            if parsed.netloc.lower() != "github.com" or not parsed.path.startswith(prefix):
                continue
            remainder = parsed.path[len(prefix) :]
            parts = remainder.split("/", 2)
            if len(parts) != 3 or parts[0] not in {"blob", "tree"}:
                continue
            kind, revision, raw_repo_path = parts
            if revision != "main":
                continue
            checked += 1
            repo_path = unquote(raw_repo_path)
            candidate = (ROOT / repo_path).resolve()
            try:
                candidate.relative_to(ROOT)
            except ValueError:
                failures.append(
                    f"{relative}: cacheon main link escapes repository: {target}"
                )
                continue
            valid = candidate.is_file() if kind == "blob" else candidate.is_dir()
            if not valid:
                expected = "file" if kind == "blob" else "directory"
                failures.append(
                    f"{relative}: cacheon main link has no local {expected}: "
                    f"{repo_path}"
                )
    return checked


def main() -> int:
    failures: list[str] = []
    paths = markdown_paths()
    redirects = check_redirect_pages(failures)
    check_private_and_retired_references(paths, failures)
    internal_links = check_internal_links(paths, failures)
    nav_pages = check_nav_coverage(redirects, failures)
    cli_commands = check_cli_inventory(failures)
    source_links = check_cacheon_main_links(paths, failures)

    if failures:
        print("Documentation checks failed:", file=sys.stderr)
        for item in failures:
            print(f"- {item}", file=sys.stderr)
        return 1

    print(
        "Documentation checks passed: "
        f"{nav_pages} nav pages, {cli_commands} CLI commands, "
        f"{internal_links} internal links, and {source_links} cacheon main links."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
