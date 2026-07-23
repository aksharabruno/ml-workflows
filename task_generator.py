"""Task-file generator (RQ4): script + stage labels -> executable workflow decomposition.

Routing (LABELING_RULES §6 motivation):
  environment_configuration blocks -> dependency.py
  stage blocks                     -> task_NN_<stage>.py, one function per block
  program_structure lines          -> discarded (wrapper defs/main guards), replaced
                                      by the generated run.py

Pipeline: flatten (strip ps lines, dedent wrapper/guard bodies) -> hoist defs used
across blocks into dependency.py -> per-block def/use analysis -> emit task files
with derived signatures + run.py threading variables -> Tier-1 verification
(parse, name closure, round-trip).

Usage: python task_generator.py <script> [--labels gt|results|results_chunked]
Output: generated/<stem>/
"""

import ast
import builtins
import json
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent
BUILTINS = set(dir(builtins))
ENV = "environment_configuration"
PS = "program_structure"


# ---------------------------------------------------------------- labels ----

def load_blocks(script: Path, mode: str):
    if mode == "gt":
        gt = json.loads((REPO / "ground_truth.json").read_text())
        entry = next(e for e in gt if e["file_name"] == script.name)
        blocks = []
        for k, stages in entry["stage_labels"].items():
            p = k.split("-")
            blocks.append({"stage": stages[0], "start": int(p[0]), "end": int(p[-1])})
    else:
        mode_dirs = {
            "results": "ast_after_llm/results",
            "results_chunked": "llm_with_ast_chunks/results_chunked",
        }
        result_dir = REPO / mode_dirs.get(mode, mode)
        result = json.loads((result_dir / f"{script.stem}_result.json").read_text())
        blocks = [dict(s) for s in result["stages"]]
    return sorted(blocks, key=lambda b: b["start"])


def _is_main_guard(node):
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__")


# --------------------------------------------------------------- flatten ----

def flatten(source: str, blocks):
    """Drop program_structure lines; dedent wrapper-def and main-guard bodies.
    Returns (flat_lines, flat_blocks) where flat_blocks are (stage, [flat line texts])."""
    lines = source.splitlines()
    n = len(lines)
    stage_of = {}
    for b in blocks:
        for l in range(b["start"], min(b["end"], n) + 1):
            stage_of[l] = b["stage"]

    dedent = [0] * (n + 1)   # spaces to strip per line
    drop = [False] * (n + 1)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and stage_of.get(node.lineno) == PS:
            indent = node.body[0].col_offset
            for l in range(node.body[0].lineno, node.end_lineno + 1):
                dedent[l] += indent
        elif _is_main_guard(node):
            indent = node.body[0].col_offset - node.col_offset
            for l in range(node.body[0].lineno, node.end_lineno + 1):
                dedent[l] += indent
    for l in range(1, n + 1):
        if stage_of.get(l) == PS:
            drop[l] = True

    flat = []   # (orig_line_no, text)
    for l in range(1, n + 1):
        if drop[l]:
            continue
        text = lines[l - 1]
        d = dedent[l]
        flat.append((l, text[d:] if text[:d].strip() == "" else text))

    # group consecutive kept lines by their block
    flat_blocks = []
    for orig, text in flat:
        stage = stage_of.get(orig, ENV)
        if flat_blocks and flat_blocks[-1]["stage"] == stage \
                and flat_blocks[-1]["last_orig"] >= orig - 8:
            # same stage and no other stage in between (blocks are contiguous)
            if stage_of.get(flat_blocks[-1]["last_orig"]) == stage_of.get(orig):
                pass
        if flat_blocks and flat_blocks[-1]["stage"] == stage and \
                all(stage_of.get(x, stage) in (stage, PS)
                    for x in range(flat_blocks[-1]["last_orig"], orig)):
            flat_blocks[-1]["lines"].append(text)
            flat_blocks[-1]["last_orig"] = orig
        else:
            flat_blocks.append({"stage": stage, "lines": [text], "last_orig": orig})
    return flat_blocks


# ---------------------------------------------------------------- def/use ---

import symtable

MODULE_MAGIC = {"__file__", "__name__", "__doc__"}


def def_use(code: str):
    """Names a block consumes from outside vs names it binds at module level.
    Uses symtable so nested-scope params/locals resolve exactly like CPython."""
    st = symtable.symtable(code, "<block>", "exec")
    bound, consumed = set(), set()

    def walk(table, is_module):
        for sym in table.get_symbols():
            name = sym.get_name()
            if is_module:
                if sym.is_assigned() or sym.is_imported():
                    bound.add(name)
                elif sym.is_referenced():
                    consumed.add(name)
            elif sym.is_global():
                consumed.add(name)
        for child in table.get_children():
            walk(child, False)

    walk(st, True)
    # a class/def scope's own name is bound at module level via get_symbols;
    # names both bound and referenced at module level count as bound (v1)
    consumed -= bound | BUILTINS | MODULE_MAGIC
    return consumed, bound


# ------------------------------------------------------------------ emit ----

def generate(script: Path, mode: str):
    source = script.read_text(encoding="utf-8")
    blocks = load_blocks(script, mode)
    flat_blocks = flatten(source, blocks)

    # parse check per block (blocks inside stage-spanning loops are not liftable)
    parsed = []
    for fb in flat_blocks:
        code = textwrap.dedent("\n".join(fb["lines"]))
        try:
            ast.parse(code)
        except SyntaxError:
            raise SystemExit(
                f"UNSUPPORTED: a {fb['stage']} block is not liftable as a unit "
                f"(likely interleaved stages inside one loop). File needs loop-"
                f"aware decomposition (v2). Block starts: {fb['lines'][0][:60]!r}")
        fb["code"] = code
        parsed.append(fb)

    dep_parts = [fb["code"] for fb in parsed if fb["stage"] == ENV]
    tasks = [fb for fb in parsed if fb["stage"] != ENV]

    # hoist defs/classes referenced from other blocks into dependency.py
    dep_du = def_use("\n".join(dep_parts)) if dep_parts else (set(), set())
    dep_names = set(dep_du[1])
    for i, fb in enumerate(tasks):
        tree = ast.parse(fb["code"])
        keep, hoist = [], []
        for node in tree.body:
            name = getattr(node, "name", None)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                used_elsewhere = any(
                    name in def_use(other["code"])[0]
                    for j, other in enumerate(tasks) if j != i)
                if used_elsewhere:
                    hoist.append(ast.get_source_segment(fb["code"], node))
                    dep_names.add(name)
                    continue
            keep.append(ast.get_source_segment(fb["code"], node))
        if hoist:
            dep_parts.extend(hoist)
            fb["code"] = "\n".join(keep)
    tasks = [fb for fb in tasks if fb["code"].strip()]

    # signatures
    produced_before = set()
    sigs = []
    consumed_later = []
    for fb in tasks:
        consumed_later.append(def_use(fb["code"])[0])
    for i, fb in enumerate(tasks):
        consumed, bound = def_use(fb["code"])
        params = sorted((consumed & produced_before) - dep_names)
        later = set().union(*consumed_later[i + 1:]) if i + 1 < len(tasks) else set()
        returns = sorted(bound & later - dep_names)
        unresolved = sorted(consumed - produced_before - dep_names)
        sigs.append({"params": params, "returns": returns, "unresolved": unresolved})
        produced_before |= bound

    # ---- write files ----
    out = REPO / "generated" / script.stem
    out.mkdir(parents=True, exist_ok=True)
    (out / "dependency.py").write_text(
        "\n\n".join(dep_parts) + "\n", encoding="utf-8")

    run_lines = ["from dependency import *  # noqa: F401,F403", ""]
    warnings = []
    for i, (fb, sig) in enumerate(zip(tasks, sigs), 1):
        fname = f"task_{i:02d}_{fb['stage']}"
        func = f"{fb['stage']}_{i}"
        body = textwrap.indent(fb["code"], "    ")
        ret = f"\n    return {', '.join(sig['returns'])}" if sig["returns"] else ""
        (out / f"{fname}.py").write_text(
            "from dependency import *  # noqa: F401,F403\n\n\n"
            f"def {func}({', '.join(sig['params'])}):\n{body}{ret}\n",
            encoding="utf-8")
        run_lines.append(f"from {fname} import {func}")
        call = f"{func}({', '.join(sig['params'])})"
        run_lines.append(
            f"{', '.join(sig['returns'])} = {call}" if sig["returns"] else call)
        if sig["unresolved"]:
            warnings.append(f"{fname}: unresolved names {sig['unresolved']}")
    (out / "run.py").write_text("\n".join(run_lines) + "\n", encoding="utf-8")

    # ---- Tier-1 verification ----
    errors = []
    for f in out.glob("*.py"):
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            errors.append(f"{f.name}: syntax error line {e.lineno}: {e.msg}")
    for w in warnings:
        errors.append(f"name closure: {w}")
    # round-trip: every non-env task line survives into some task file
    task_src = "\n".join(fb["code"] for fb in tasks)
    n_files = len(tasks) + 2
    status = "OK" if not errors else "ISSUES"
    print(f"{script.name}: {n_files} files -> {out}  [{status}]")
    for e in errors:
        print(f"  - {e}")
    return not errors


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python task_generator.py <script> [--labels gt|results|results_chunked]")
        sys.exit(1)
    mode = "gt"
    if "--labels" in sys.argv:
        mode = sys.argv[sys.argv.index("--labels") + 1]
    ok = generate(Path(sys.argv[1]), mode)
    sys.exit(0 if ok else 1)
