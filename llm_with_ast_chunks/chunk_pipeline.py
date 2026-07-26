"""Chunk-classification pipeline variant ("Way B" / ablation arm).

AST chops the script into logical units (ast_chunker); the LLM assigns a stage
per chunk and never emits a line number. Imports, main guards, docstrings and
glue-only chunks are labeled mechanically; def/try headers are derived from
their children (uniform -> that label, mixed -> program_structure).

Usage:
  python chunk_pipeline.py <path_to_file>   # classify a single script
  python chunk_pipeline.py --all            # classify every file in ground_truth.json
Results land in results_chunked/<stem>_result.json (same schema as main.py,
scoreable via: python evaluate.py llm_with_ast_chunks/results_chunked).
The --all sweep only runs files that have a ground_truth.json entry, so it never
touches a script whose ground truth has not been written yet (R30).
"""

import json
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# shared modules at repo root; JSON parser lives in the range-mode package
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "ast_after_llm"))

from ast_chunker import chunk_source, resolve_labels, expand_to_lines, lines_to_stages, dump_tree
from llm_detector import _parse_json_response
from prompts import build_chunk_prompt

load_dotenv()

VALID = {
    "environment_configuration", "data_preparation", "feature_engineering",
    "model_generation", "model_evaluation", "program_structure",
}




def run(input_path: Path, quiet: bool = False):
    source = input_path.read_text(encoding="utf-8")
    chunks = chunk_source(source)
    if not quiet:
        print(dump_tree(chunks))                # pre-labeling view; debugging

    prompt = build_chunk_prompt(source, chunks)
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    text_block = next((b for b in response.content if b.type == "text"), None)
    if text_block is None:
        raise RuntimeError("no text block in LLM response")
    parsed = _parse_json_response(text_block.text, verbose=not quiet)
    if not parsed:
        raise RuntimeError("could not parse LLM response")

    raw = parsed.get("chunk_labels", {})
    llm_labels = {int(k): v for k, v in raw.items()
                  if isinstance(v, str) and v in VALID}

    if not quiet:
        print(llm_labels)  # raw LLM output; debugging

    resolved = resolve_labels(chunks, llm_labels)
    line_label = expand_to_lines(source, chunks, resolved)
    stages = lines_to_stages(line_label)

    if not quiet:
        print(dump_tree(chunks, labels=resolved))  # final stages; debugging

    result = {
        "file": input_path.name,
        "is_ml_training_workflow": parsed.get("is_ml_training_workflow", False),
        "ml_problem": parsed.get("ml_problem", "unknown"),
        "stages": stages,
        "mode": "chunk_classification",
        "n_chunks": len(chunks),
        "n_llm_chunks": sum(1 for c in chunks if c["llm"]),
    }
    out_dir = Path(__file__).resolve().parent / "results_chunked"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{input_path.stem}_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"{input_path.name}: {len(stages)} blocks, "
          f"{result['n_llm_chunks']}/{result['n_chunks']} chunks LLM-labeled "
          f"-> {out_path.name}")
    return result


def corpus_files():
    """Files with a ground_truth.json entry (R30: GT written before we run)."""
    gt = json.loads((_REPO_ROOT / "ground_truth.json").read_text())
    test_dir = _REPO_ROOT / "test_data"
    for entry in gt:
        name = entry.get("file_name")
        if not name:
            continue
        path = test_dir / name
        if path.exists():
            yield path
        else:
            print(f"[skip] {name}: not found in test_data/")


def run_all():
    files = list(corpus_files())
    print(f"Running chunk pipeline on {len(files)} file(s)...\n")
    ok = failed = 0
    for path in files:
        try:
            run(path, quiet=True)
            ok += 1
        except Exception as exc:  # keep going; one bad file shouldn't stop the sweep
            print(f"{path.name}: FAILED — {exc}")
            failed += 1
    print(f"\nDone: {ok} ok, {failed} failed.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage:")
        print("  python chunk_pipeline.py <path_to_file>   # one file")
        print("  python chunk_pipeline.py --all            # every file in ground_truth.json")
        sys.exit(1)

    if args[0] in ("--all", "-a"):
        run_all()
    else:
        p = Path(args[0])
        if not p.exists():
            print(f"File not found: {p}")
            sys.exit(1)
        run(p)
