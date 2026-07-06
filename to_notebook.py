"""
Convert an analyzed ML script into a Jupyter notebook, one code cell per
LLM-detected stage block (from results/<stem>_result.json).

Each code cell is preceded by a markdown cell naming the stage, and carries
the stage name in its metadata tags for programmatic consumers.

Usage: python to_notebook.py <script.py> [output.ipynb]
"""

import json
import sys
from pathlib import Path

import nbformat

RESULTS_DIR = Path("results")
DEFAULT_OUTPUT_DIR = Path("notebooks")


def _clamped_blocks(stages: list, n_lines: int) -> list[dict]:
    """Sort blocks, clamp to file length, and close any gaps so every
    source line lands in exactly one cell (gap lines join the earlier block,
    leading lines join the first block)."""
    blocks = sorted(
        ({**b, "end": min(b["end"], n_lines)} for b in stages if b["start"] <= n_lines),
        key=lambda b: b["start"],
    )
    if not blocks:
        return [{"stage": "unlabeled", "start": 1, "end": n_lines}]

    blocks[0]["start"] = 1
    for prev, cur in zip(blocks, blocks[1:]):
        prev["end"] = cur["start"] - 1
    blocks[-1]["end"] = n_lines
    return blocks


def _title(stage: str) -> str:
    return stage.replace("_", " ").title()


def convert(source_file: Path, result_json: Path, output_path: Path):
    result = json.loads(result_json.read_text(encoding="utf-8"))
    stages = result.get("stages", [])
    if not stages:
        print("No stage blocks in result — run main.py first.")
        sys.exit(1)

    source_lines = source_file.read_text(encoding="utf-8").splitlines()
    blocks = _clamped_blocks(stages, len(source_lines))

    nb = nbformat.v4.new_notebook()
    nb.metadata["ml_workflows"] = {
        "source_file": source_file.name,
        "ml_problem": result.get("ml_problem", "unknown"),
        "is_ml_training_workflow": result.get("is_ml_training_workflow", False),
    }

    header = (
        f"# {source_file.name}\n\n"
        f"ML problem: **{result.get('ml_problem', 'unknown')}**"
    )
    if result.get("reasoning"):
        header += f"\n\n{result['reasoning']}"
    nb.cells.append(nbformat.v4.new_markdown_cell(header))

    for block in blocks:
        code = "\n".join(source_lines[block["start"] - 1 : block["end"]]).strip("\n")
        if not code.strip():
            continue
        nb.cells.append(
            nbformat.v4.new_markdown_cell(f"## {_title(block['stage'])}")
        )
        cell = nbformat.v4.new_code_cell(code)
        cell.metadata["tags"] = [block["stage"]]
        cell.metadata["ml_workflows"] = {
            "stage": block["stage"],
            "source_lines": [block["start"], block["end"]],
        }
        nb.cells.append(cell)

    nbformat.validate(nb)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, output_path)
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"Written: {output_path}  ({n_code} stage cells)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python to_notebook.py <script.py> [output.ipynb]")
        sys.exit(1)

    source = Path(sys.argv[1])
    result = RESULTS_DIR / f"{source.stem}_result.json"
    output = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else DEFAULT_OUTPUT_DIR / f"{source.stem}.ipynb"
    )

    if not result.exists():
        print(f"No result found at {result} — run main.py first.")
        sys.exit(1)

    convert(source, result, output)
