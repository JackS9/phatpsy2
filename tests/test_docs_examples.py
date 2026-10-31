# tests/test_docs_examples.py
"""
Automated test suite verifying executable documentation examples in docs/.
Prevents documentation drift by:
1. Parsing and loading all YAML calculation decks via CalculationDeck.from_yaml().
2. AST-parsing all Python code snippets and resolving phatpsy.* imports against the codebase.
3. Verifying documented CLI flags against phatpsy.cli argument parsers.
"""
from pathlib import Path
import re
import ast
import textwrap
import yaml
import pytest

import phatpsy
from phatpsy.schema import CalculationDeck


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"


def get_doc_files():
    """Retrieve all markdown documentation files."""
    return sorted(list(DOCS_DIR.rglob("*.md")))


# ==============================================================================
# 1. YAML Calculation Decks Verification
# ==============================================================================

def test_docs_yaml_blocks():
    """Verify that all YAML blocks in documentation are syntactically valid,
    and all complete decks instantiate CalculationDeck without error."""
    yaml_blocks_checked = 0
    decks_checked = 0

    for md_file in get_doc_files():
        content = md_file.read_text(encoding="utf-8")
        matches = re.finditer(r"```(?:yaml|yml)\n(.*?)```", content, re.DOTALL)
        for idx, m in enumerate(matches):
            raw_code = m.group(1)
            # Skip blocks explicitly tagged to skip
            if "<!-- test: skip -->" in content[max(0, m.start()-50):m.start()]:
                continue

            yaml_blocks_checked += 1
            try:
                parsed = yaml.safe_load(raw_code)
            except Exception as exc:
                pytest.fail(f"Invalid YAML in {md_file.relative_to(REPO_ROOT)} block {idx}: {exc}")

            # If it defines a complete CalculationDeck (contains geometry or title)
            if isinstance(parsed, dict) and ("geometry" in parsed or "title" in parsed):
                try:
                    deck = CalculationDeck.from_yaml(raw_code)
                    assert deck.geometry is not None or deck.title is not None
                    decks_checked += 1
                except Exception as exc:
                    pytest.fail(
                        f"CalculationDeck.from_yaml failed in {md_file.relative_to(REPO_ROOT)} block {idx}: {exc}"
                    )

    assert yaml_blocks_checked > 0, "No YAML blocks discovered in docs/"
    assert decks_checked > 0, "No CalculationDeck YAML examples discovered in docs/"


# ==============================================================================
# 2. Python Code Snippets Verification
# ==============================================================================

def test_docs_python_blocks():
    """Verify that all Python code blocks in documentation are valid Python (AST parse)
    and all imported phatpsy attributes exist in the installed package."""
    py_blocks_checked = 0
    symbols_checked = 0

    for md_file in get_doc_files():
        content = md_file.read_text(encoding="utf-8")
        matches = re.finditer(r"```python\n(.*?)```", content, re.DOTALL)
        for idx, m in enumerate(matches):
            raw_code = m.group(1)
            # Skip blocks explicitly tagged to skip
            if "<!-- test: skip -->" in content[max(0, m.start()-50):m.start()]:
                continue

            code = textwrap.dedent(raw_code)
            try:
                tree = ast.parse(code)
            except Exception as exc:
                pytest.fail(f"Python SyntaxError in {md_file.relative_to(REPO_ROOT)} block {idx}: {exc}")

            py_blocks_checked += 1

            # Traverse AST to verify imported phatpsy symbols
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module == "phatpsy":
                        for alias in node.names:
                            symbols_checked += 1
                            if not hasattr(phatpsy, alias.name):
                                pytest.fail(
                                    f"Undefined symbol 'phatpsy.{alias.name}' documented in "
                                    f"{md_file.relative_to(REPO_ROOT)} block {idx}"
                                )
                    elif node.module and node.module.startswith("phatpsy."):
                        submod_name = node.module.split(".")[1]
                        if hasattr(phatpsy, submod_name):
                            submod = getattr(phatpsy, submod_name)
                            for alias in node.names:
                                symbols_checked += 1
                                if not hasattr(submod, alias.name):
                                    pytest.fail(
                                        f"Undefined symbol '{node.module}.{alias.name}' documented in "
                                        f"{md_file.relative_to(REPO_ROOT)} block {idx}"
                                    )

    assert py_blocks_checked > 0, "No Python blocks discovered in docs/"
    assert symbols_checked > 0, "No phatpsy symbols verified in docs/"


# ==============================================================================
# 3. CLI Command and Flag Verification
# ==============================================================================

def test_docs_cli_flags():
    """Verify that documented CLI invocations in bash/console code blocks
    use valid options corresponding to their phatpsy subcommand."""
    # Mapping of known subcommands to allowed options (including short and long)
    # Extracted from phatpsy.cli argument parsers
    subcommand_options = {
        "cube": {
            "-h", "--help", "--mo", "--ao", "--density", "--spin-density",
            "--deformation", "--esp", "--isovalue", "--spacing", "--padding",
            "-o", "--output", "--view", "-n", "--no-browser",
        },
        "supercell": {
            "-h", "--help", "-e", "--element", "-c", "--crystal", "-a",
            "--lattice-constant", "--bulk", "-m", "--miller", "-l", "--layers",
            "-v", "--vacuum", "-r", "--repeat", "-f", "--fixed-layers",
            "--virtual-perimeter", "-s", "--auto-spin", "--adsorbate",
            "--site", "--height", "--tilt", "--angle", "-b", "--basis",
            "--format", "-o", "--output", "--view", "-n", "--no-browser",
        },
        "slab": {
            "-h", "--help", "-e", "--element", "-c", "--crystal", "-a",
            "--lattice-constant", "--bulk", "-m", "--miller", "-l", "--layers",
            "-v", "--vacuum", "-r", "--repeat", "-f", "--fixed-layers",
            "--virtual-perimeter", "-s", "--auto-spin", "--adsorbate",
            "--site", "--height", "--tilt", "--angle", "-b", "--basis",
            "--format", "-o", "--output", "--view", "-n", "--no-browser",
        },
        "elements": {
            "-h", "--help", "-b", "--basis",
        },
        "build": {
            "-h", "--help", "-b", "--basis", "-u", "--units", "-t", "--title",
            "-s", "--auto-spin", "-o", "--output",
        },
        "scan": {
            "-h", "--help", "-b", "--bond", "-a", "--angle", "-c", "--coord",
            "-r", "--range", "--bond2", "--coord2", "--range2", "-o", "--output",
            "-p", "--plot", "--no-warm-restart",
        },
        "opt": {
            "-h", "--help", "-o", "--output", "-t", "--trajectory", "-p", "--plot",
            "--max-steps", "--grad-tol", "--energy-tol", "--no-warm-restart",
        },
        "vib": {
            "-h", "--help", "-s", "--step", "-m", "--method", "-T", "--temp",
            "-P", "--pressure", "--no-project", "-o", "--output", "--view",
        },
        "md": {
            "-h", "--help", "-n", "--steps", "--dt", "-T", "--temp", "--thermostat",
            "--tau", "--no-com-removal", "--no-warm-restart", "-o", "--output",
            "--xyz", "-p", "--plot", "--view",
        },
        "neb": {
            "-h", "--help", "-n", "--images", "-k", "--spring",
            "--max-iter", "--force-tol", "--no-climb", "--no-warm-restart",
            "-o", "--output", "--xyz", "-p", "--plot", "--view",
        },
        "dimer": {
            "-h", "--help", "-d", "--dimer-dist", "-n", "--max-cycles",
            "--force-tol", "--rot-tol", "-o", "--output", "--xyz", "--view",
        },
        "irc": {
            "-h", "--help", "-s", "--step", "-n", "--max-steps",
            "--direction", "--grad-tol", "--diff-step", "-o", "--output",
            "--xyz", "-p", "--plot", "--view",
        },
        "view": {
            "-h", "--help", "-o", "--output", "-s", "--auto-spin",
            "-e", "--edit", "-n", "--no-browser",
        },
        "convert": {
            "-h", "--help", "-o", "--output",
        },
        "chemisorption": {
            "-h", "--help", "--site", "--c4v", "--atom", "--sites",
            "--lattice", "--vib", "--scls", "--all", "-o", "--output",
            "--json-stdout",
        },
        "eom": {
            "-h", "--help", "-s", "--spacing", "-T", "--temp",
            "--probe-wavelengths", "-o", "--output", "-p", "--plot",
            "--html", "--view",
        },
        "root": {
            "-h", "--help", "-o", "--output", "--json-stdout",
        },
    }
    # Register aliases
    subcommand_options["pes"] = subcommand_options["scan"]
    subcommand_options["optimize"] = subcommand_options["opt"]
    subcommand_options["freq"] = subcommand_options["vib"]
    subcommand_options["vibrations"] = subcommand_options["vib"]
    subcommand_options["bomd"] = subcommand_options["md"]
    subcommand_options["dynamics"] = subcommand_options["md"]
    subcommand_options["ci-neb"] = subcommand_options["neb"]
    subcommand_options["path"] = subcommand_options["neb"]
    subcommand_options["grid"] = subcommand_options["cube"]
    subcommand_options["volumetric"] = subcommand_options["cube"]
    subcommand_options["properties"] = subcommand_options["eom"]
    subcommand_options["ads"] = subcommand_options["chemisorption"]
    subcommand_options["chemi"] = subcommand_options["chemisorption"]

    commands_checked = 0
    for md_file in get_doc_files():
        content = md_file.read_text(encoding="utf-8")
        matches = re.finditer(r"```(?:bash|sh|console)\n(.*?)```", content, re.DOTALL)
        for idx, m in enumerate(matches):
            for line in m.group(1).splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Strip inline comment
                line = re.sub(r"\s+#.*$", "", line).strip()
                tokens = line.split()
                if not tokens:
                    continue

                # Locate phatpsy invocation
                phatpsy_idx = -1
                for i, tok in enumerate(tokens):
                    if tok == "phatpsy" or tok.endswith("/phatpsy"):
                        phatpsy_idx = i
                        break
                if phatpsy_idx == -1:
                    continue

                args = tokens[phatpsy_idx + 1:]
                if not args:
                    continue

                subcmd = "root"
                if args and not args[0].startswith("-"):
                    if args[0] in subcommand_options:
                        subcmd = args[0]
                        args = args[1:]

                allowed = subcommand_options[subcmd]
                commands_checked += 1

                for arg in args:
                    if arg.startswith("-") and arg != "-":
                        # Extract flag name (strip =value if present)
                        flag = arg.split("=")[0]
                        if flag not in allowed:
                            pytest.fail(
                                f"Unknown flag '{flag}' for subcommand '{subcmd}' documented in "
                                f"{md_file.relative_to(REPO_ROOT)}: '{line}'"
                            )

    assert commands_checked > 0, "No CLI commands verified in docs/"


# ==============================================================================
# 4. Documentation Prose Style and Lint Verification
# ==============================================================================

def test_docs_style_lint():
    """Verify that all markdown documentation files satisfy prose lint standards:
    no GitHub-style alerts, no untagged code blocks, and no out-of-tree links."""
    from tools.lint_docs import lint_docs
    res = lint_docs()
    assert res == 0, "Documentation style lint failed; see stderr for details."
