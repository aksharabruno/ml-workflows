"""AST-based chunker for the chunk-classification pipeline variant ("Way B").

Chops a script into logical units with exact line spans so the LLM only
assigns a stage per chunk and never emits a line number. Conventions mirror
LABELING_RULES.md:

- import statements                  -> auto environment_configuration (R10)
- `if __name__ == "__main__":` line  -> auto program_structure (R28)
- def/async-def header lines         -> DERIVED after labeling: all direct
                                        children share one label -> that label
                                        (R24); mixed -> program_structure (R27)
- class defs                         -> one atomic chunk, labeled by LLM (R24)
- other compound headers (for/while/ -> merged into their first body chunk
  with/try/if, incl. except/else)      (R8: one logical unit)
- docstring expressions              -> glue, attach forward (R1 analogy)
- blanks/comments (not in any span)  -> glue fill: blanks backward (R3),
                                        comments forward (R1), trailing
                                        backward (R4)

Compound statements are recursed into only when their span is >= RECURSE_MIN
lines; smaller ones stay atomic (R8).
"""

import ast

RECURSE_MIN = 8

ENV = "environment_configuration"
PS = "program_structure"


def _is_main_guard(node) -> bool:
    if not isinstance(node, ast.If):
        return False
    t = node.test
    return (
        isinstance(t, ast.Compare)
        and isinstance(t.left, ast.Name)
        and t.left.id == "__name__"
        and len(t.comparators) == 1
        and isinstance(t.comparators[0], ast.Constant)
        and t.comparators[0].value in ("__main__",)
    )


def _is_docstring_stmt(node) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _node_start(node) -> int:
    """Start line including decorators."""
    decs = getattr(node, "decorator_list", [])
    if decs:
        return min(d.lineno for d in decs)
    return node.lineno


class Chunker:
    def __init__(self, source: str):
        self.tree = ast.parse(source)
        self.n_lines = len(source.splitlines())
        self.chunks = []  # dicts: id,start,end,kind,auto_label,parent,llm

    def run(self):
        self._chunk_body(self.tree.body, parent=None)
        self.chunks.sort(key=lambda c: c["start"])
        for i, c in enumerate(self.chunks, 1):
            c["id"] = i
        return self.chunks

    def _add(self, start, end, kind, parent, auto_label=None):
        self.chunks.append({
            "id": None, "start": start, "end": end, "kind": kind,
            "auto_label": auto_label, "parent": parent,
            "llm": auto_label is None and kind not in ("def_header", "docstring"),
        })
        return self.chunks[-1]

    def _chunk_body(self, body, parent, prepend_line=None):
        """Chunk a list of statements. prepend_line: a compound-header line
        to merge into the first chunk produced (R8 header treatment)."""
        first_extra = prepend_line
        for node in body:
            start = _node_start(node)
            if first_extra is not None:
                start = min(start, first_extra)
            produced_start = start
            first_extra_used = first_extra
            first_extra = None

            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._add(produced_start, node.end_lineno, "import", parent, ENV)
            elif _is_docstring_stmt(node):
                self._add(produced_start, node.end_lineno, "docstring", parent)
            elif _is_main_guard(node):
                self._add(produced_start, node.lineno, "main_guard", parent, PS)
                self._chunk_body(node.body, parent)
                if node.orelse:
                    self._chunk_body(node.orelse, parent)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                header = self._add(
                    produced_start, node.body[0].lineno - 1, "def_header", parent
                )
                self._chunk_body(node.body, header)
            elif isinstance(node, ast.ClassDef):
                self._add(produced_start, node.end_lineno, "class", parent)
            elif (
                isinstance(node, ast.Try)
                and node.end_lineno - node.lineno + 1 >= RECURSE_MIN
            ):
                # try: header derives like a def header — uniform body -> that
                # label, mixed body -> program_structure (LABELING_RULES §7)
                header = self._add(produced_start, node.lineno, "def_header", parent)
                self._chunk_body(node.body, header)
                for handler in node.handlers:
                    self._chunk_body(handler.body, header,
                                     prepend_line=handler.lineno)
                if node.orelse:
                    self._chunk_body(node.orelse, header,
                                     prepend_line=node.orelse[0].lineno)
                if node.finalbody:
                    self._chunk_body(node.finalbody, header,
                                     prepend_line=node.finalbody[0].lineno)
            elif (
                isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.If,
                                  ast.With, ast.AsyncWith))
                and node.end_lineno - node.lineno + 1 >= RECURSE_MIN
            ):
                # merge the compound header line(s) into the first body chunk
                self._chunk_body(node.body, parent, prepend_line=produced_start)
                orelse = getattr(node, "orelse", None)
                if orelse:
                    self._chunk_body(orelse, parent,
                                     prepend_line=orelse[0].lineno)
            else:
                self._add(produced_start, node.end_lineno, "stmt", parent)

            # if prepend never got consumed (empty body), widen nothing
            del first_extra_used


def _is_glue_line(line: str) -> bool:
    # keep aligned with evaluate.is_glue
    s = line.strip()
    return not s or s.startswith("#") or s.startswith("print(") or s.startswith("print ")


def chunk_source(source: str):
    chunks = Chunker(source).run()
    # chunks made entirely of glue (bare prints/comments/blanks) are not
    # labelable content — treat like docstrings: glue-attach, never LLM-labeled,
    # excluded from def-header derivation
    lines = source.splitlines()
    for c in chunks:
        if c["kind"] in ("stmt",) and all(
            _is_glue_line(lines[l - 1])
            for l in range(c["start"], min(c["end"], len(lines)) + 1)
        ):
            c["kind"] = "glue"
            c["llm"] = False
    return chunks


def resolve_labels(chunks, llm_labels: dict):
    """Combine auto labels, LLM labels, and derived def-header labels.
    llm_labels: {chunk_id(int or str): stage}. Returns {chunk_id: stage}."""
    labels = {}
    for c in chunks:
        if c["auto_label"]:
            labels[c["id"]] = c["auto_label"]
        elif c["llm"]:
            lab = llm_labels.get(c["id"]) or llm_labels.get(str(c["id"]))
            labels[c["id"]] = lab or "data_preparation"  # safe fallback

    # docstrings + glue-only chunks: attach forward to the next labeled chunk
    ordered = sorted(chunks, key=lambda c: c["start"])
    for i, c in enumerate(ordered):
        if c["kind"] in ("docstring", "glue"):
            nxt = next((labels.get(d["id"]) for d in ordered[i + 1:]
                        if d["id"] in labels), None)
            prv = next((labels.get(d["id"]) for d in reversed(ordered[:i])
                        if d["id"] in labels), None)
            labels[c["id"]] = nxt or prv or ENV

    # derive def headers bottom-up (deepest spans first)
    headers = [c for c in chunks if c["kind"] == "def_header"]

    def derive(h):
        kids = [labels[c["id"]] for c in chunks
                if c["parent"] is h and c["id"] in labels
                and c["kind"] not in ("docstring", "glue")]
        if not kids:
            return PS
        return kids[0] if len(set(kids)) == 1 else PS

    # two passes so nested headers propagate outward (innermost first)
    for _ in range(2):
        for h in sorted(headers, key=lambda c: c["end"] - c["start"]):
            labels[h["id"]] = derive(h)
    return labels


def expand_to_lines(source: str, chunks, labels: dict):
    """Produce a full line->stage map (1..N): chunk lines take their chunk's
    label; glue lines fill per R1/R3/R4."""
    lines = source.splitlines()
    n = len(lines)
    line_label = {}
    for c in sorted(chunks, key=lambda c: c["start"]):
        lab = labels[c["id"]]
        for l in range(c["start"], min(c["end"], n) + 1):
            line_label.setdefault(l, lab)

    def next_labeled(i):
        for j in range(i + 1, n + 1):
            if j in line_label:
                return line_label[j]
        return None

    def prev_labeled(i):
        for j in range(i - 1, 0, -1):
            if j in line_label:
                return line_label[j]
        return None

    for i in range(1, n + 1):
        if i in line_label:
            continue
        blank = not lines[i - 1].strip()
        lab = prev_labeled(i) if blank else (next_labeled(i) or prev_labeled(i))
        line_label[i] = lab or ENV

    return line_label


def lines_to_stages(line_label: dict):
    """Merge consecutive same-label lines into ordered stage blocks."""
    stages = []
    for i in sorted(line_label):
        lab = line_label[i]
        if stages and stages[-1]["stage"] == lab and stages[-1]["end"] == i - 1:
            stages[-1]["end"] = i
        else:
            stages.append({"stage": lab, "start": i, "end": i})
    return stages
