# Detection and Extraction of ML Training Workflows

A Python tool that detects and analyzes machine-learning training workflows in Python code. An LLM segments a script into pipeline stages; targeted AST analysis then extracts structured facts (dataset, model, hyperparameters) from the regions the LLM identified.

## Architecture

```
script.py ──> LLM (stage detection) ──> stages + ml_problem ──> AST extraction ──> final JSON
              claude-opus-4-8            {stage, start, end}     dataset / model /    results/<stem>_result.json
              5 stage definitions        per line-range          hyperparameters
```

1. **LLM layer** (`llm_detector.py`, `prompts.py`) — one API call per file. The source is embedded in the prompt with explicit line numbers (`N:` prefixes) so the model reads positions instead of counting them. Returns contiguous stage blocks covering every line, plus the ML problem type and whether the file is a training workflow at all.
2. **AST layer** (`ast_parser.py`) — parses the source and mines the LLM-identified regions: dataset load calls from `data_preparation` blocks; model instantiation and hyperparameters from `model_generation` blocks (resolved against `stage_map.py`).
3. **Orchestrator** (`main.py`) — runs both layers and writes the combined result.

## Stage taxonomy

Five stages, aligned with the four AutoML pipeline phases consolidated in Rajenthiram et al. 2025 (*AutoML: A Tertiary Study of Phases, Methods, Tools, and Frameworks*), plus one code-level stage for imports:

| Stage | Covers |
|---|---|
| `environment_configuration` | imports, library/logging/device setup |
| `data_preparation` | loading, cleaning, EDA, scaling/normalization, encoding (incl. target labels), train/test splitting, DataLoaders |
| `feature_engineering` | feature selection (incl. manual column picking), construction, extraction (PCA, tokenization/vectorization, embeddings) |
| `model_generation` | model/architecture definition, compile/fit, hand-rolled training loops, hyperparameter search, model saving |
| `model_evaluation` | predictions for scoring, metrics, validation on held-out data |

Boundary conventions follow the paper: scaling/encoding is data preparation (not feature engineering); the generation/evaluation boundary is where training ends and scoring begins.

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Create a `.env` file with your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

Analyze a script:

```bash
./venv/bin/python main.py test_data/script_t1_03.py
# -> results/script_t1_03_result.json
```

Evaluate against ground truth:

```bash
python3 evaluate.py
```

Convert an analyzed script to a stage-segmented Jupyter notebook (one code cell per stage block, stage name in cell tags/metadata — e.g. for CellFlow visualization):

```bash
./venv/bin/python to_notebook.py test_data/script_t1_03.py
# -> notebooks/script_t1_03.ipynb
```

## Result format

```json
{
  "file": "script_t1_03.py",
  "is_ml_training_workflow": true,
  "ml_problem": "regression",
  "dataset": {"source": null, "load_call": "read_csv"},
  "model": {"name": "train", "library": "lightgbm", "full_call": "lightgbm.train"},
  "hyperparameters": [{"name": "train", "call": "lightgbm.train", "params": {"num_boost_round": 20}}],
  "stages": [
    {"stage": "environment_configuration", "start": 1, "end": 7},
    {"stage": "data_preparation", "start": 8, "end": 22},
    {"stage": "model_generation", "start": 23, "end": 46},
    {"stage": "model_evaluation", "start": 47, "end": 52}
  ],
  "reasoning": "..."
}
```

## Test data

`test_data/` contains scripts and notebooks at three complexity levels:

1. **Level 1 — Syntactic baseline**: standard library calls and variable names (sklearn/XGBoost/LightGBM/Keras tutorials).
2. **Level 2 — Semantic alias**: standard libraries, non-standard structure — wrapper functions, custom `nn.Module` classes, hand-rolled training loops, interleaved stages, out-of-execution-order definitions.
3. **Level 3 — Architectural shift**: no standard "fit"-style keywords.

## Ground truth & evaluation

`ground_truth_v2.json` holds per-file labels: stage line ranges, `ml_problem`, `models`, `is_ml_training_workflow`.

Labeling conventions:
- Blocks are contiguous and gap-free; every line belongs to exactly one block.
- A line belongs to the stage that consumes its output (a params dict used by `fit` is `model_generation`).
- Comments and status prints attach forward to the stage they introduce; result-reporting prints attach backward.
- Blank lines belong to the same block as the nearest preceding non-blank line (they close the block above; a block never starts blank).

`evaluate.py` scores per **line × stage** against the ground truth, restricted to lines the ground truth labels. Glue lines (blank, comment-only, bare `print`) are **masked** — excluded from scoring — so boundary placement inside glue never affects the metrics. Ranges are clamped to the actual file length. The script also reports file-level accuracy for `ml_problem` and `is_ml_training_workflow`.

Current results (9 scripts, levels 1–2): micro F1 = 0.91; all Level-1 files at 1.00; residual disagreements on Level 2 are open labeling-convention questions (training-history plots, tokenization stage, logging setup), not model errors.

## Repository layout

| Path | Purpose |
|---|---|
| `main.py` | pipeline orchestrator |
| `llm_detector.py` | LLM call + robust JSON parsing |
| `prompts.py` | stage definitions and prompt builder |
| `ast_parser.py` | AST extraction from LLM-identified regions |
| `stage_map.py` | fully-qualified call → fine-grained stage lookup |
| `evaluate.py` | per-line evaluation against ground truth |
| `to_notebook.py` | script → stage-segmented .ipynb converter |
| `ground_truth_v2.json` | human stage labels (current, 5-stage) |
| `test_data/` | test scripts and notebooks (3 levels) |
| `results/` | pipeline outputs |
| `notebooks/` | generated stage-segmented notebooks |
| `splitter.py`, `headergen_parser.py`, `ground_truth.json` | legacy (pre-pivot pipeline) |

## Roadmap

- Notebook (`.ipynb`) input support via cell-marker normalization
- Evaluation of the AST extraction layer (dataset/model/hyperparameters); metrics extraction from `model_evaluation` regions
- Level-3 scripts and notebook ground truth
- Integration with [CellFlow](https://github.com/selincoban95/CellFlow) for in-JupyterLab stage visualization
