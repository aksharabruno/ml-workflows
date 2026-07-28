"""Chunk-classification pipeline: script -> stage-labeled results (+ optional task files).

Flow (see also the llm_labeler / ast_chunker / task_generator modules):
  ast_chunker  splits the script into logical units
  llm_labeler  asks the LLM to label each chunk (never emits line numbers)
  ast_chunker  resolves / derives / merges the labels into line-level stages
  -> results/<stem>_result.json
  With --decompose, task_generator then decomposes the script into executable
  task files under generated/<stem>/.

Usage:
  python pipeline.py <path>              # classify one script
  python pipeline.py <path> --decompose   # classify + generate task files
  python pipeline.py --all               # classify every file in ground_truth.json
  python pipeline.py --all --decompose    # ... + generate task files for each
Score results with: python evaluate.py results
The --all sweep only runs files that have a ground_truth.json entry, so it never
touches a script whose ground truth has not been written yet (R30).
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

from ast_chunker import chunk_source, resolve_labels, expand_to_lines, lines_to_stages, dump_tree
from llm_labeler import label_chunks
import task_generator

RESULTS_DIR = _REPO_ROOT / "results"


def run(input_path: Path, quiet: bool = False, decompose: bool = False):
    source = input_path.read_text(encoding="utf-8")
    chunks = chunk_source(source)
    if not quiet:
        print(dump_tree(chunks))                # pre-labeling view; debugging

    # --- LLM labeling (llm_labeler owns the model interaction) ---
    parsed, llm_labels = label_chunks(source, chunks, quiet=quiet)
    if not quiet:
        print(llm_labels)                       # raw LLM output; debugging

    # --- resolve/derive/merge into line-level stages (ast_chunker) ---
    resolved = resolve_labels(chunks, llm_labels)
    stages = lines_to_stages(expand_to_lines(source, chunks, resolved))
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
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{input_path.stem}_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"{input_path.name}: {len(stages)} blocks, "
          f"{result['n_llm_chunks']}/{result['n_chunks']} chunks LLM-labeled "
          f"-> {out_path.name}")

    # --- optional downstream: decompose into task files ---
    if decompose:
        try:
            task_generator.generate(input_path, mode=str(RESULTS_DIR))
        except (Exception, SystemExit) as exc:
            # a decomposition failure (UNSUPPORTED / etc.) must not void the
            # results we already wrote, nor stop a batch run.
            print(f"{input_path.name}: decomposition skipped — {exc}")

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


def run_all(decompose: bool = False):
    files = list(corpus_files())
    suffix = " (+ task files)" if decompose else ""
    print(f"Running chunk pipeline on {len(files)} file(s){suffix}...\n")
    ok = failed = 0
    for path in files:
        try:
            # plain --all stays extraction-only (benchmark-friendly);
            # --all --decompose also decomposes each file (fail-safe per file).
            run(path, quiet=True, decompose=decompose)
            ok += 1
        except Exception as exc:  # keep going; one bad file shouldn't stop the sweep
            print(f"{path.name}: FAILED — {exc}")
            failed += 1
    print(f"\nDone: {ok} ok, {failed} failed.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage:")
        print("  python pipeline.py <path>            # one file")
        print("  python pipeline.py <path> --decompose # one file + task files")
        print("  python pipeline.py --all             # every file in ground_truth.json")
        print("  python pipeline.py --all --decompose  # every file + task files")
        sys.exit(1)

    if args[0] in ("--all", "-a"):
        run_all(decompose="--decompose" in args)
    else:
        p = Path(args[0])
        if not p.exists():
            print(f"File not found: {p}")
            sys.exit(1)
        run(p, decompose="--decompose" in args)
