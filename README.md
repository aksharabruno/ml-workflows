# Detection and Extraction of ML Training Workflows

Real-world ML scripts are written flat: data loading, training, and evaluation
interleaved in one file, with the workflow structure existing only in the author's
head. This tool recovers that structure — labeling every line with its pipeline
stage — and uses it to decompose the script into an executable, per-stage task
workflow.

## The pipeline

One design principle runs through it: **deterministic code owns geometry and facts;
the LLM answers only questions of meaning.** Python's own parser decides where units
begin and end; the model decides only what they are, so positional errors are
impossible by construction.

```
script.py ─> AST chunker ─> LLM (label per chunk) ─> resolve/derive/merge ─> results JSON ─> task decomposition
             logical units   ~70% of chunks           rule + LLM + derived    results/       (optional, --decompose)
             exact spans     (rest auto/derived)       labels
```

1. **`ast_chunker`** splits the script into logical units (most are single
   statements). Imports and main guards are labeled by rule; `def`/`try` headers are
   derived from their children (uniform → that label, mixed → `program_structure`);
   glue attaches by fixed conventions.
2. **`llm_labeler`** sends the chunked script to the LLM, which assigns one stage per
   chunk and **never emits a line number**.
3. **`ast_chunker`** resolves/derives/merges those chunk labels back into per-line
   stages.
4. **`pipeline`** writes the result JSON — and, with `--decompose`, hands it to
   **`task_generator`** to produce the executable task files.

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

All boundary conventions are codified as a numbered, dated rule set with case law in
[LABELING_RULES.md](LABELING_RULES.md) — the annotator guide that makes the ground
truth reproducible rather than one person's opinion.

## Setup

```bash
python3 -m venv venv                          # Python 3.10+ (uses PEP 604 `X | None` syntax)
./venv/bin/pip install -r requirements.txt    # pipeline deps only (anthropic, dotenv, nbformat)
```

`requirements-corpus.txt` holds the corpus scripts' own dependencies — install it only
to execute generated task workflows, not for analysis.

Create a `.env` file with your Anthropic API key: `ANTHROPIC_API_KEY=sk-ant-...`

## Usage

```bash
# Classify a single script  -> results/<stem>_result.json
python pipeline.py test_data/script_t1_01.py

# Classify + decompose into task files
python pipeline.py test_data/script_t1_01.py --decompose

# Classify every file that has a ground-truth entry (R30-safe)
python pipeline.py --all
python pipeline.py --all --decompose          # ... and decompose each

# Score results against ground truth
python evaluate.py results

# Decompose from a chosen label source (gt | results | a results dir)
python task_generator.py test_data/script_t1_01.py --labels gt

# Script -> stage-segmented notebook (for CellFlow visualization)
python to_notebook.py test_data/script_t1_01.py
```

Plain `--all` is extraction-only (benchmark-friendly); add `--decompose` for the full
extract-then-decompose sweep. Decomposition is fail-safe per file — a script that
cannot be decomposed is reported and skipped, never crashing the run.

## Task decomposition

Stage labels form a routing table. `task_generator` flattens the script (strips
`program_structure` scaffolding, dedents wrapper/guard bodies), then emits one folder
per task following the ExtremeXP experimentation-engine layout:

```
generated/<stem>/
  dependency.py                       # imports, constants, shared helper defs
  task_01_data_preparation/
    task_01_data_preparation.py       # block wrapped in a function
  task_02_model_generation/
    task_02_model_generation.py
  ...
  run.py                              # calls tasks in order, threading variables
```

Function signatures come from AST def/use analysis (consumed variables → parameters,
variables needed later → returns); `run.py` wires them so the reassembled workflow
runs. Built-in verification checks that every artifact parses and every consumed name
is bound. Most files decompose fully; the rest are **detected, not silently
mis-generated** — the two failure modes are multi-stage helper functions (whose
definition is scaffolding) and stages interleaved inside a single compound statement
(a loop or `with` context). Generated workflows are exactly as runnable as their
source, including failing identically on missing data or local imports.

## Corpus & ground truth

`test_data/` holds real-world scripts at three complexity levels — Level 1 linear
tutorials, Level 2 structural complexity (wrapper functions, custom classes,
hand-rolled loops, interleaving), Level 3 real-world scale and noise (cloud APIs,
tracking, notebook exports, delegation to local modules) — plus inference-only and
non-ML negatives that test workflow-vs-non-workflow detection only. Level 2–3 files
are sourced from post-cutoff GitHub repositories with verifiable provenance to limit
training-data contamination (rules R30/R35). The corpus currently holds 28 scripts
and is being expanded toward 30 positive workflows (10 per level) + 10 negatives.

`ground_truth.json` holds per-file labels: stage line ranges (every line covered
exactly once), `ml_problem` (`classification-binary` / `classification-multiclass` /
`regression`), `models`, `is_ml_training_workflow`, `level`, `source_url`. Ground
truth is written **before** the pipeline ever runs on a file (process rule R30).

## Evaluation

`evaluate.py` scores per **line × stage**, restricted to ground-truth-covered lines.
Glue lines (blanks, comments, bare prints) are masked so boundary placement inside
glue never moves the metrics. Negatives count toward workflow detection only and are
excluded from line scoring. It also reports `ml_problem` accuracy (exact subtype and
coarse) and `is_ml_training_workflow`.

Because there is no prior tool doing line-level stage extraction to compare against,
**per-stage F1 is the meaningful lens** — how well each stage is identified — rather
than a single overall number.

Pilot (single run; the N=5 mean±sd benchmark on a frozen corpus is the citable
protocol, pending):

| Metric | Value |
|---|---|
| micro F1 (per line × stage) | ~0.93 |
| `ml_problem` (subtype + coarse) | 21/23 scored files — the 2 misses are dual-task scripts outside the single-label schema |
| strongest stages | `model_generation`, `environment_configuration`, `data_preparation` |
| weakest stages | `program_structure` (over-emission on inference/UI code), `feature_engineering` (tokenization routed to data_prep) |

Residual errors concentrate in documented open conventions (see LABELING_RULES.md),
not random failure.

## Repository layout

| Path | Purpose |
|---|---|
| `pipeline.py` | orchestration + CLI (single file / `--all`, optional `--decompose`) |
| `ast_chunker.py` | splits script into chunks; resolves/derives/merges labels into line-level stages |
| `llm_labeler.py` | the LLM layer: prompt call, response parsing, per-chunk labels |
| `prompts.py` | stage definitions + chunk-labeling prompt |
| `task_generator.py` | labeled script → per-task folders + `dependency.py` + `run.py` (verified) |
| `evaluate.py` | scores results against ground truth (per line × stage, per-stage F1) |
| `to_notebook.py` | script → stage-segmented `.ipynb` |
| `ground_truth.json` | human stage labels + metadata for the corpus |
| `LABELING_RULES.md` | the numbered annotator guide (+ printable .docx/.pdf) |
| `test_data/` | the corpus |
| `results/` | pipeline outputs |
| `generated/` | task-file decompositions |
| `deprecated/` | superseded code (former range-mode pipeline) |

## Roadmap

- **N=5 benchmark** (mean±sd) on the frozen corpus; per-stage F1 breakdown and
  line-level mismatch analysis (no external baseline — no prior line-level extractor
  exists)
- **Corpus completion** to 30 positives (10/10/10) + 10 negatives (inference-only, non-ML)
- **Semantic preservation** verified manually on a decomposable subset; hyperparameter
  extraction via AST (secondary)
- **ExtremeXP alignment**: mapping generated decompositions onto the ExtremeXP
  task/workflow model (conceptual — no DSL emission); CellFlow in-JupyterLab
  stage visualization
- **Open taxonomy question**: `model_application` (inference/serving code) — deferred
  pending benchmark error analysis
- **Future work**: whole-project (multi-file) decomposition via import resolution and
  call-site stage inheritance
