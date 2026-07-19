"""Chunk-classification pipeline variant ("Way B" / ablation arm).

AST chops the script into logical units (ast_chunker); the LLM assigns a stage
per chunk and never emits a line number. Imports, main guards, docstrings and
glue-only chunks are labeled mechanically; def/try headers are derived from
their children (uniform -> that label, mixed -> program_structure).

Usage: python chunk_pipeline.py <path_to_file>
Results land in results_chunked/<stem>_result.json (same schema as main.py,
scoreable via: python evaluate.py results_chunked).
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
from prompts import STAGE_DEFINITIONS

load_dotenv()

VALID = {
    "environment_configuration", "data_preparation", "feature_engineering",
    "model_generation", "model_evaluation", "program_structure",
}


def build_chunk_prompt(source: str, chunks) -> str:
    lines = source.splitlines()
    parts = []
    for c in chunks:
        text = "\n".join(lines[c["start"] - 1:c["end"]])
        if c["llm"]:
            tag = f'CHUNK {c["id"]} (lines {c["start"]}-{c["end"]}) — LABEL THIS:'
        elif c["auto_label"]:
            tag = (f'chunk {c["id"]} (lines {c["start"]}-{c["end"]}) — '
                   f'already labeled: {c["auto_label"]}')
        else:
            tag = (f'chunk {c["id"]} (lines {c["start"]}-{c["end"]}) — '
                   f'derived automatically, do not label')
        parts.append(f"### {tag}\n{text}")
    chunk_listing = "\n\n".join(parts)
    need = [str(c["id"]) for c in chunks if c["llm"]]

    return f"""
You are an ML pipeline stage classifier. The script below has been pre-split
into chunks by a parser. Assign exactly one stage to every chunk marked
"LABEL THIS". Do not invent line numbers — chunk boundaries are fixed.

## The 6 valid stage labels (use ONLY these exact strings):
- "environment_configuration"
- "data_preparation"
- "feature_engineering"
- "model_generation"
- "model_evaluation"
- "program_structure"

## Stage definitions:
{STAGE_DEFINITIONS}

## Labeling rules:
- A chunk that computes something belongs to the stage that CONSUMES its
  output (a params dict used by a fit call -> model_generation; a path used by
  read_csv -> data_preparation; `num_labels = len(set(labels))` used by the
  model constructor -> model_generation). Multiple consumers -> FIRST consumer.
- Constants/config blocks whose values feed several different stages ->
  environment_configuration. A dict consumed by a single call follows data
  flow to that call.
- Training-loop chunks: per-epoch validation, early stopping, checkpointing,
  and training-accuracy tracking inside the loop are model_generation. Only
  post-training scoring on held-out data is model_evaluation. Post-fit
  decision-threshold selection on VALIDATION data is tuning -> model_generation;
  diagnostics reported on training data (OOB, train-set confusion) ->
  model_evaluation.
- Functions/classes are labeled by their CONTENT, not their call site.
  Metric helper defs -> model_evaluation. Training-history plots after
  training -> model_evaluation.
- Use the whole script for context, not chunks in isolation.

## The script, in chunks:
{chunk_listing}

## Output
"ml_problem" must be one of: "classification", "regression", "clustering", "other".
Return ONLY a JSON object, no markdown, no explanation:
{{
    "is_ml_training_workflow": true,
    "ml_problem": "classification",
    "chunk_labels": {{ {", ".join(f'"{i}": "<stage>"' for i in need[:3])}, ... }}
}}
"chunk_labels" must contain every chunk id marked LABEL THIS: {", ".join(need)}
"""


def run(input_path: Path):
    source = input_path.read_text(encoding="utf-8")
    chunks = chunk_source(source)
    print(dump_tree(chunks))                    # pre-labeling view (shown above); debugging
    

    prompt = build_chunk_prompt(source, chunks)
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    text_block = next((b for b in response.content if b.type == "text"), None)
    if text_block is None:
        print("No text block in response")
        sys.exit(1)
    parsed = _parse_json_response(text_block.text, verbose=True)
    if not parsed:
        print("Could not parse LLM response")
        sys.exit(1)

    raw = parsed.get("chunk_labels", {})
    llm_labels = {int(k): v for k, v in raw.items()
                  if isinstance(v, str) and v in VALID}

    print(llm_labels)  # raw LLM output; debugging

    resolved = resolve_labels(chunks, llm_labels)
    line_label = expand_to_lines(source, chunks, resolved)
    stages = lines_to_stages(line_label)

    print(dump_tree(chunks, labels=resolve_labels(chunks, llm_labels)))  # final stages; debugging

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
          f"{result['n_llm_chunks']}/{result['n_chunks']} chunks LLM-labeled")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chunk_pipeline.py <path_to_file>")
        sys.exit(1)
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(1)
    run(p)
