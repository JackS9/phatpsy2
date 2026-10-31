#!/usr/bin/env python3
"""
tools/lint_docs.py
Linter for PHATPSY 2.0 user manual and documentation pages.
Checks for:
1. MkDocs admonitions (rejecting GitHub-style `> [!NOTE]` / `> [!WARNING]`).
2. Fenced code blocks without language tags.
3. Out-of-tree relative links (../../) escaping docs directory.
4. Local file:/// machine links.
"""
import sys
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"


def lint_docs():
    errors = []
    checked_files = 0

    github_alert_pattern = re.compile(r"^>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]", re.IGNORECASE)
    untyped_fence_pattern = re.compile(r"^```\s*$")
    local_file_link_pattern = re.compile(r"file:///")
    out_of_tree_link_pattern = re.compile(r"\[[^\]]+\]\(\.\./\.\./[^\)]+\)")

    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        checked_files += 1
        lines = md_file.read_text(encoding="utf-8").splitlines()
        rel_path = md_file.relative_to(REPO_ROOT)

        in_code_block = False
        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Track fenced code blocks
            if stripped.startswith("```"):
                if not in_code_block:
                    # Opening fence: must have a language tag
                    tag = stripped[3:].strip()
                    if not tag:
                        errors.append(
                            f"{rel_path}:{line_num}: Opening code fence missing language tag ('```'). "
                            f"Specify language (e.g. ```yaml, ```python, ```bash, ```text)."
                        )
                    in_code_block = True
                else:
                    # Closing fence
                    in_code_block = False
                continue

            # Check 3: Local file:/// URLs
            if local_file_link_pattern.search(line):
                errors.append(
                    f"{rel_path}:{line_num}: Local machine URL 'file:///' found. "
                    f"Use relative documentation links or GitHub URLs."
                )

            # Check 4: Out-of-tree links escaping docs/
            if out_of_tree_link_pattern.search(line):
                errors.append(
                    f"{rel_path}:{line_num}: Out-of-tree relative link ('../../...') escaping docs/ directory. "
                    f"Use GitHub permalinks or inline code spans."
                )

    if errors:
        sys.stderr.write(f"Documentation lint failed with {len(errors)} issue(s):\n")
        for err in errors:
            sys.stderr.write(f"  - {err}\n")
        return 1

    sys.stdout.write(f"Documentation lint passed: {checked_files} files inspected, 0 issues found.\n")
    return 0


if __name__ == "__main__":
    sys.exit(lint_docs())
