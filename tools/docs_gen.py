"""MkDocs build hook: generate reference pages straight from the source.

Run by the ``mkdocs-gen-files`` plugin (see ``mkdocs.yml``). Everything is
derived statically with ``ast`` so the docs build never imports ``phatpsy``
(which needs the compiled ``_phatpsy_core`` extension) and cannot drift from
the code:

* ``user/yaml/reference.md`` -- every YAML key, from the dataclasses in
  ``phatpsy/schema.py`` (type, default, trailing ``# comment``, docstring).
* ``user/cli/reference.md``  -- every subcommand and option, from the
  ``argparse`` calls in ``phatpsy/cli.py``.
* ``user/python/reference.md`` -- mkdocstrings directives for the public API
  listed in ``phatpsy/__init__.py::__all__``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "phatpsy"

SCHEMA_CLASSES = [
    ("CalculationDeck", "Top level of a deck"),
    ("ControlConfig", "`control:` block"),
    ("AtomConfig", "Each entry of `geometry:`"),
    ("SymmetrySpec", "Legacy `symmetry:` form of an atom"),
    ("ElementConfig", "Each entry of `elements:`"),
    ("BasisFunction", "Each entry of an element's `functions:`"),
    ("InitialOrbital", "Each entry of an atom's `starting_orbitals:`"),
]
# Source files searched (in order) for the classes above.
SCHEMA_FILES = ["schema.py", "basis.py"]


def _esc(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


# --------------------------------------------------------------------------
# YAML reference
# --------------------------------------------------------------------------
def _field_default(node: ast.AnnAssign) -> str:
    if node.value is None:
        return "*required*"
    v = node.value
    if (
        isinstance(v, ast.Call)
        and getattr(v.func, "id", None) == "field"
        and v.keywords
    ):
        kw = {k.arg: k.value for k in v.keywords}
        if "default_factory" in kw:
            fac = kw["default_factory"]
            if isinstance(fac, ast.Lambda):
                return f"`{ast.unparse(fac.body)}`"
            return f"`{ast.unparse(fac)}()`"
        if "default" in kw:
            return f"`{ast.unparse(kw['default'])}`"
    return f"`{ast.unparse(v)}`"


def build_yaml_reference() -> str:
    classes = {}  # name -> (ClassDef, source lines)
    for fname in SCHEMA_FILES:
        src = (PKG / fname).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for n in tree.body:
            if isinstance(n, ast.ClassDef):
                classes.setdefault(n.name, (n, src.splitlines()))

    out = [
        "# YAML key reference",
        "",
        "!!! note \"Generated\"",
        "    This page is generated from the dataclasses in "
        "[`phatpsy/schema.py`](https://github.com/JackS9/phatpsy2/blob/master/phatpsy/schema.py) "
        "at build time, so it always matches the code.",
        "",
    ]
    for name, blurb in SCHEMA_CLASSES:
        if name not in classes:
            continue
        cls, lines = classes[name]
        out += [f"## `{name}` — {blurb}", ""]
        doc = ast.get_docstring(cls)
        if doc:
            out += [doc, ""]
        rows = []
        for node in cls.body:
            if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
                continue
            comment = ""
            line = lines[node.lineno - 1]
            if "#" in line:
                comment = line.split("#", 1)[1].strip()
            rows.append(
                f"| `{node.target.id}` | `{_esc(ast.unparse(node.annotation))}` "
                f"| {_field_default(node)} | {_esc(comment)} |"
            )
        if rows:
            out += ["| Key | Type | Default | Notes |", "| --- | --- | --- | --- |", *rows, ""]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# CLI reference
# --------------------------------------------------------------------------
class _CliVisitor(ast.NodeVisitor):
    """Collect each ArgumentParser and its add_argument calls in source order."""

    def __init__(self) -> None:
        self.parsers: list[dict] = []
        self._aliases: list[str] = []

    @staticmethod
    def _consts(node: ast.AST) -> list[str]:
        if isinstance(node, ast.Compare):
            for comp in node.comparators:
                if isinstance(comp, (ast.Tuple, ast.List)):
                    return [e.value for e in comp.elts if isinstance(e, ast.Constant)]
                if isinstance(comp, ast.Constant):
                    return [comp.value]
        return []

    def visit_If(self, node: ast.If) -> None:
        self._aliases = self._consts(node.test)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "ArgumentParser":
            kw = {k.arg: k.value for k in node.keywords}
            prog = ast.literal_eval(kw["prog"]) if "prog" in kw else "phatpsy"
            desc = ast.literal_eval(kw["description"]) if "description" in kw else ""
            self.parsers.append(
                {"prog": prog, "desc": desc, "aliases": list(self._aliases), "args": []}
            )
        elif isinstance(fn, ast.Attribute) and fn.attr == "add_argument" and self.parsers:
            names = [a.value for a in node.args if isinstance(a, ast.Constant)]
            kw = {}
            for k in node.keywords:
                try:
                    kw[k.arg] = ast.literal_eval(k.value)
                except Exception:
                    kw[k.arg] = ast.unparse(k.value)
            if "type" in {k.arg for k in node.keywords}:
                kw["type"] = ast.unparse(next(k.value for k in node.keywords if k.arg == "type"))
            self.parsers[-1]["args"].append((names, kw))
        self.generic_visit(node)


def build_cli_reference() -> str:
    tree = ast.parse((PKG / "cli.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    v = _CliVisitor()
    v.visit(main)

    out = [
        "# CLI reference",
        "",
        "!!! note \"Generated\"",
        "    Generated from the `argparse` definitions in "
        "[`phatpsy/cli.py`](https://github.com/JackS9/phatpsy2/blob/master/phatpsy/cli.py).",
        "",
    ]
    for p in v.parsers:
        is_default = p["prog"] == "phatpsy"
        out += [f"## `{p['prog']}`" + (" (default run)" if is_default else ""), ""]
        if p["aliases"]:
            alts = [a for a in p["aliases"] if a != p["prog"].split()[-1]]
            if alts:
                out += ["Aliases: " + ", ".join(f"`phatpsy {a}`" for a in alts), ""]
        if p["desc"]:
            out += [p["desc"], ""]
        rows = []
        for names, kw in p["args"]:
            flag = ", ".join(f"`{n}`" for n in names)
            if kw.get("action") == "store_true":
                typ = "flag"
            elif "choices" in kw:
                typ = "one of " + ", ".join(f"`{c}`" for c in kw["choices"])
            else:
                typ = f"`{kw.get('type', 'str')}`"
            default = kw.get("default")
            default = "—" if default in (None, False) and kw.get("action") == "store_true" else (
                "*none*" if default is None else f"`{default}`"
            )
            if kw.get("required"):
                default = "*required*"
            elif not names[0].startswith("-") and kw.get("nargs") != "?":
                default = "*required*" if kw.get("default") is None else default
            rows.append(f"| {flag} | {typ} | {default} | {_esc(str(kw.get('help', '')))} |")
        if rows:
            out += ["| Argument | Type | Default | Description |", "| --- | --- | --- | --- |", *rows, ""]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# Python API reference (mkdocstrings)
# --------------------------------------------------------------------------
# __all__ symbol -> defining module (resolved statically from __init__.py imports)
def build_api_reference() -> str:
    tree = ast.parse((PKG / "__init__.py").read_text(encoding="utf-8"))
    where: dict[str, str] = {}
    all_names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("phatpsy."):
            for alias in node.names:
                where[alias.asname or alias.name] = node.module
        elif isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "__all__" for t in node.targets
        ):
            all_names = [e.value for e in node.value.elts]

    by_module: dict[str, list[str]] = {}
    for name in all_names:
        mod = where.get(name)
        if mod:
            by_module.setdefault(mod, []).append(name)

    out = [
        "# Python API reference",
        "",
        "!!! note \"Generated\"",
        "    Built by mkdocstrings from the docstrings and signatures of every symbol in "
        "`phatpsy.__all__`, grouped by defining module.",
        "",
    ]
    for mod in sorted(by_module):
        out += [f"## `{mod}`", ""]
        for name in by_module[mod]:
            out += [f"::: {mod}.{name}", "    options:", "      heading_level: 3", ""]
    return "\n".join(out) + "\n"


for rel, text in (
    ("user/yaml/reference.md", build_yaml_reference()),
    ("user/cli/reference.md", build_cli_reference()),
    ("user/python/reference.md", build_api_reference()),
):
    with mkdocs_gen_files.open(rel, "w") as fh:
        fh.write(text)
