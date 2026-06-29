"""
Split an ML script into per-stage task files based on LLM stage labels.
Output files are named task_<stage>.py and placed in a subdirectory.
"""

import json
import re
import sys
from pathlib import Path


def parse_line_ranges(stage_labels: dict) -> list[tuple[int, int, list[str]]]:
    """Return sorted list of (start, end, [stages]) from LLM stage_labels."""
    ranges = []
    for key, stages in stage_labels.items():
        parts = key.split("-")
        start = int(parts[0])
        end = int(parts[1]) if len(parts) > 1 else start
        ranges.append((start, end, stages))
    return sorted(ranges, key=lambda x: x[0])


def extract_imports(source_lines: list[str]) -> list[str]:
    """Return all import lines from the source."""
    imports = []
    for line in source_lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(line.rstrip())
    return imports


def split_script(result_json: Path, source_file: Path, output_dir: Path):
    with open(result_json) as f:
        result = json.load(f)

    stage_labels = result.get("llm_analysis", {}).get("stage_labels", {})
    if not stage_labels:
        print("No LLM stage labels found.")
        return

    source_lines = source_file.read_text(encoding="utf-8").splitlines()
    ranges = parse_line_ranges(stage_labels)
    imports = extract_imports(source_lines)

    # Map each line number (1-indexed) to its primary stage
    line_to_stage: dict[int, str] = {}
    for start, end, stages in ranges:
        stage = stages[0]  # use first stage if multiple
        if stage == "library_loading":
            continue  # imports handled separately
        for lineno in range(start, end + 1):
            if lineno not in line_to_stage:
                line_to_stage[lineno] = stage

    # Group consecutive lines by stage, preserving order
    stage_blocks: dict[str, list[str]] = {}
    seen_order: list[str] = []
    for lineno in sorted(line_to_stage):
        stage = line_to_stage[lineno]
        line = source_lines[lineno - 1]  # convert to 0-indexed
        if stage not in stage_blocks:
            stage_blocks[stage] = []
            seen_order.append(stage)
        stage_blocks[stage].append(line)

    if not stage_blocks:
        print("No stages to split.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for stage in seen_order:
        lines = stage_blocks[stage]
        filename = output_dir / f"task_{stage}.py"

        content_parts = []
        if imports:
            content_parts.append("\n".join(imports))
            content_parts.append("")  # blank line
        content_parts.append(f"# Stage: {stage}")
        content_parts.append("\n".join(lines))

        filename.write_text("\n".join(content_parts) + "\n", encoding="utf-8")
        print(f"  Written: {filename.name}  ({len(lines)} lines)")

    print(f"\nSplit into {len(seen_order)} task files in {output_dir}/")
    print(f"Stage order: {' -> '.join(seen_order)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python splitter.py <script.py>")
        sys.exit(1)

    source = Path(sys.argv[1])
    result = Path("results") / f"{source.stem}_result.json"
    out = Path("task_output") / source.stem

    if not result.exists():
        print(f"No result found at {result} — run main.py first.")
        sys.exit(1)

    split_script(result, source, out)
