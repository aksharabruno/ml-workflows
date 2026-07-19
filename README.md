# Detection and Extraction of ML Training Workflows

Real-world ML scripts are written flat: data loading, training, and evaluation
interleaved in one file, with the workflow structure existing only in the author's
head. This tool recovers that structure — labeling every line with its pipeline
stage — and uses it to decompose the script into an executable, per-stage task
workflow.

## Two pipeline modes

Both modes share one design principle: **deterministic code owns geometry and facts;
the LLM answers only questions of meaning.** Python's own parser decides where units
begin and end; the model decides what they are.

**Range mode** (`ast_after_llm/`) — the LLM segments a line-numbered script into
contiguous stage blocks; targeted AST analysis then extracts structured facts
(dataset, model, hyperparameters) from the regions the LLM identified.

```
script.py ─> LLM (stage ranges) ─> AST extraction ─> results JSON
             claude-opus-4-8       dataset/model/     ast_after_llm/results/
             numbered lines        hyperparameters
```

**Chunk mode** (`llm_with_ast_chunks/`) — the AST splits the script into logical
units first (86% single statements); the LLM only assigns a stage per chunk and
never emits a line number, making positional errors impossible by construction.
Imports and main guards are labeled by rule; def/try headers are derived from their
children (uniform → that label, mixed → `program_structure`); glue attaches by fixed
conventions; block boundaries come from the parser.

```
script.py ─> AST chunker ─> LLM (label per chunk) ─> resolve/derive/merge ─> results JSON
             logical units   ~70% of chunks           rule + LLM + derived    llm_with_ast_chunks/
             exact spans     (rest auto/derived)      labels                  results_chunked/
```

## Stage taxonomy

Six labels: four ML stages following the AutoML phase consolidation of Rajenthiram
et al. 2025 (*AutoML: A Tertiary Study of Phases, Methods, Tools, and Frameworks*),
plus two code-level labels required for total line coverage of real code.

| Stage | Covers |
|---|---|
| `environment_configuration` | imports, device/seed/logging setup, argparse & constants blocks, tracking setup, pretrained processing artifacts |
| `data_preparation` | loading, cleaning, EDA, scaling/encoding, transform pipelines (incl. augmentation), splitting, DataLoaders |
| `feature_engineering` | feature selection/construction/extraction, tokenization/vectorization, embedding matrices |
| `model_generation` | model definition, training (incl. in-loop validation & threshold tuning), hyperparameter search, checkpoint save/load |
| `model_evaluation` | post-training scoring, metrics, diagnostics, history plots, reporting |
| `program_structure` | lines whose syntactic scope spans multiple stages: multi-stage wrapper defs, main guards, multi-stage try wrappers |

All boundary conventions are codified as 34 numbered rules with case law in
[LABELING_RULES.md](LABELING_RULES.md) — the annotator guide that makes the ground
truth reproducible rather than one person's opinion.

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt   # pipeline deps only (anthropic, dotenv, nbformat)
```

`requirements-corpus.txt` holds the corpus scripts' own dependencies — install it only
to execute generated task workflows (Tier-2/3 verification), not for analysis.

Create a `.env` file with your Anthropic API key: `ANTHROPIC_API_KEY=sk-ant-...`

## Usage

```bash
# Range mode
./venv/bin/python ast_after_llm/main.py test_data/script_t1_03.py

# Chunk mode
./venv/bin/python llm_with_ast_chunks/chunk_pipeline.py test_data/script_t1_03.py

# Evaluate either mode against ground truth
python3 ast_after_llm/evaluate.py                                      # range results
python3 ast_after_llm/evaluate.py llm_with_ast_chunks/results_chunked  # chunk results

# Decompose a script into task files (labels from gt | results | results_chunked)
python3 task_generator.py test_data/script_t1_01.py --labels gt
# -> generated/script_t1_01/: dependency.py, task_NN_<stage>.py, run.py

# Script -> stage-segmented notebook (for CellFlow visualization)
./venv/bin/python to_notebook.py test_data/script_t1_03.py
```

## Task-file generation

Stage labels form a routing table: `environment_configuration` → `dependency.py`,
each stage block → `task_NN_<stage>.py` wrapping the block in a function whose
signature is derived by AST def/use analysis (consumed variables → parameters,
variables needed later → returns), `program_structure` → discarded and replaced by a
generated `run.py` that threads the variables. Built-in Tier-1 verification checks
parseability and name closure. 19/22 corpus files decompose; refusals are detected
(stages interleaved inside one compound statement), not silently mis-generated.
Generated workflows are exactly as runnable as their source scripts, including
failing identically on missing data or local imports.

## Corpus & ground truth

`test_data/` holds 22 real-world scripts at three complexity levels — Level 1 linear
tutorials, Level 2 structural complexity (wrapper functions, custom classes,
hand-rolled loops, interleaving), Level 3 real-world scale and noise (cloud APIs,
tracking, notebook exports, delegation to local modules) — plus one negative
(inference-only). Level 2–3 files were sourced from post-cutoff GitHub repositories
to avoid training-data contamination.

`ground_truth.json` holds per-file labels: stage line ranges (every line covered
exactly once), `ml_problem`, `models`, `is_ml_training_workflow`, `level`,
`source_url`. Ground truth is written before the pipeline ever runs on a file
(process rule R30).

## Evaluation

`evaluate.py` scores per **line × stage**, restricted to ground-truth-covered lines.
Glue lines (blanks, comments, bare prints) are masked so boundary placement inside
glue never moves the metrics. Also reports per-file `ml_problem` and
`is_ml_training_workflow` accuracy.

Pilot results (single run per mode — the multi-run benchmark with mean±sd is the
citable protocol, pending):

| | micro F1 | notes |
|---|---|---|
| range mode | 0.91 | full 22-file corpus |
| chunk mode | 0.94 | oracle ceiling 1.0000 (chunking loses nothing) |
| `ml_problem` / workflow detection | 22/22 | both modes |

Residual errors concentrate in documented open conventions (see LABELING_RULES.md §7),
not random failure.

## Repository layout

| Path | Purpose |
|---|---|
| `ast_after_llm/` | range-mode pipeline: `main.py`, `llm_detector.py`, `ast_parser.py`, `evaluate.py`, `results/` |
| `llm_with_ast_chunks/` | chunk-mode pipeline: `ast_chunker.py`, `chunk_pipeline.py`, `results_chunked/` |
| `prompts.py` | shared stage definitions + range-mode prompt |
| `stage_map.py` | call→stage lookup (extraction layer + naive baseline) |
| `task_generator.py` | labels → dependency.py + task files + run.py (Tier-1 verified) |
| `to_notebook.py` | script → stage-segmented .ipynb |
| `ground_truth.json` | human stage labels for all 22 files |
| `LABELING_RULES.md` | the 34-rule annotator guide (+ printable .docx/.pdf) |
| `test_data/` | the corpus |
| `generated/` | task-file decompositions |
| `temp/`, `splitter.py` | legacy (pre-pivot) |

## Roadmap

- Multi-run benchmark (mean±sd, both modes) on the frozen corpus; baselines
  (keyword STAGE_MAP; HeaderGen) and ablations (line numbering, chunk vs range,
  model swap, 5 vs 6 labels, glue masking, RECURSE_MIN sweep)
- Negative-example expansion (ML-without-training, non-ML) scored on workflow
  detection only
- ExtremeXP alignment: mapping of generated decompositions onto the ExtremeXP
  task/workflow model (conceptual — no DSL emission); CellFlow in-JupyterLab stage
  visualization
- Open taxonomy question: `model_application` (inference/serving code) — deliberately
  deferred pending benchmark error analysis
- Chunk-mode "MIXED" escape hatch for sub-threshold compounds spanning stages
