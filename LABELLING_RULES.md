# Ground-Truth Labeling Rules

Annotator guide for `ground_truth.json`. Every line of a script receives exactly one stage label. These rules exist so that two annotators labeling the same file independently produce the same result.

Taxonomy alignment: the four ML stages follow the AutoML phase consolidation of Rajenthiram et al. 2025 (Figure 2 and section 4.3 settles scaling/encoding/augmentation boundaries); the two code-level labels (`environment_configuration`, `program_structure`) are our extensions, required for total line coverage of real code.

---

## 1. The labels

| Label                       | One-line definition                                                                       |
| --------------------------- | ----------------------------------------------------------------------------------------- |
| `environment_configuration` | Sets up the conditions for computation; no data, features, or model flow through it       |
| `data_preparation`          | Loading, cleaning, exploring (EDA), and transforming data into model-ready form           |
| `feature_engineering`       | Selecting, constructing, or extracting features to improve the input representation       |
| `model_generation`          | Defining the model, training it, tuning it, saving it                                     |
| `model_evaluation`          | Assessing the trained model on held-out data: predictions for scoring, metrics, reporting |
| `program_structure`         | Lines whose syntactic scope spans multiple stages (wrapper defs, main guards)             |

A file may omit any ML stage. A training script with no held-out evaluation has no `model_evaluation` block (see R21). A file may also be a negative: `is_ml_training_workflow: false` for inference-only or non-ML code.

---

## 2. Core decision procedure

Apply in order:

**Step 1 - Anchors.** Lines that directly perform a stage-defining action take that stage's label: `read_csv` → data_prep, `PCA(...)` → feature_eng, `model.fit(...)` → model_gen, `accuracy_score(...)` → model_eval, `import` → env_config.

**Step 2 - Data flow.** Every other line that computes something belongs to the stage that **consumes its output**: a `params` dict used by `lgb.train` → model_generation; a path variable used by `read_csv` → data_preparation; `num_labels = len(set(labels))` used by the model constructor → model_generation. If output is consumed by multiple stages, use the **first** consumer.

**Step 3 - Glue.** Lines that compute nothing (blanks, comments, bare prints):

- **R1.** Comments and status prints attach **forward** to the block they introduce (`# train`, `print("Loading data...")`).
- **R2.** Result-reporting prints attach **backward** to the stage that produced the value (`print(f"RMSE: {rmse}")` → model_evaluation). This is Step 2 applied to prints.
- **R3.** **Blank lines attach backward**: a blank line belongs to the same block as the nearest preceding non-blank line. A block never starts with a blank line.
- **R4.** Trailing glue at end of file attaches backward (there is no next block).

Glue placement never affects evaluation scores (blanks/comments/bare prints are masked)— these rules exist for consistency and readability only.

---

## 3. Block mechanics

- **R5.** Blocks are contiguous and gap-free: every line 1..N belongs to exactly one block.
- **R6.** Blocks are per-BLOCK, not per-stage: a stage may appear multiple times (interleaving is normal at Levels 2–3). Never merge same-stage blocks across an intervening different-stage block — that would misrepresent execution order.
- **R7.** Single-line blocks are legal (`"73": "environment_configuration"` for a lone `hvd.init()`).
- **R8.** Do not split a logical unit (a loop body, a function body, a multi-line call) across two blocks unless it genuinely spans stages.
- **R9.** Ranges must not overshoot the file's line count.

---

## 4. environment_configuration — what counts

Litmus test: _would this line survive unchanged if you swapped in a different dataset and model?_ Yes → env_config.

- **R10.** Imports; runtime library config (`gdal.UseExceptions()`, `warnings.filterwarnings`).
- **R11.** Hardware/compute setup: device selection, `cuda.is_available()`, `set_device`. But moving a model/tensor to a device (`model.to(device)`) → the model/tensor's stage.
- **R12.** Distributed runtime init (`hvd.init()`), but broadcasts of model/optimizer state → model_generation (state flows through them).
- **R13.** Random seeds, even mid-file.
- **R14.** CLI/config parsing and literal-constant blocks → env_config. The whole argparse block (with arg unpacking) and Config-class unpacking (BATCH_SIZE = Config.BATCH_SIZE) stay together as one env block; likewise a comment-introduced block of literal constants that feed multiple stages (BATCH_SIZE = 32, EPOCHS = 10, paths). Rationale: these values feed many stages, so applying data-flow would shred a cohesive block. Distinction: values computed from data (N, D = X.shape, input_dim = X_train.shape[1]) are not constants, they follow data flow (Step 2). The cohesion rule covers literal config constants only.
- **R15.** Experiment-tracking _setup_ (TensorBoard writer, logdir creation, MLflow init). _Using_ the tracker (`run.log(x)`) → the stage that produced x.
- **R16.** Version asserts and compatibility checks.
- Plot _styling_ (`sns.set`, `rcParams`) is NOT env_config → it serves the plotting stage (usually data_preparation EDA). Global runtime config → env; stage-serving config → that stage.
- **R17.** Loading a pretrained _processing artifact_ (tokenizer, vectorizer) via `from_pretrained`/hub fetch → **env_config** (provisioning: an import-with-arguments, analogous to R10 and auto-`pip install`; also routes to dependency.py in task generation). 

---

## 5. Boundary rulings

- **R18.** Scaling/normalization/encoding, including of targets (`to_categorical(y)`, `LabelEncoder` on y) and format conversions (reshape, astype, `torch.tensor(x)`) → **data_preparation** (paper Fig. 2/ Sec. 4.3). NOT feature engineering.
- **R19.** Image/tensor transform pipelines → **data_preparation**, including _augmentation_ (random flips/crops, mixup, ColorJitter). Rationale: follows the paper (paper Fig. 2, our taxonomy's cited authority), and real code interleaves format and augmentation transforms inside a single `Compose([...])` call, which R8 forbids splitting.
- **R20.** Feature engineering = selection (including manual column picking, `X = df[cols]`), construction, extraction (PCA, tokenization/vectorization, embedding-matrix construction). Text tokenization → feature_engineering.
- **R21.** In-loop validation (per-epoch val phase inside the training loop) → **model_generation** (training monitoring). Only held-out scoring after training is model_evaluation. Consequence: a script may legitimately have no model_evaluation block. Training-accuracy tracking inside the train phase is always model_generation. **Extended:** after training ends, the criterion is what the result _feeds_: computation whose output becomes part of the model/decision rule → model_generation (post-fit decision-threshold selection on validation, t2_07 132-145- the chosen threshold joins the predictor, so it is tuning); computation whose output is only _reported_ → model_evaluation (training-set diagnostics: OOB score, feature importances, train-set confusion- t3_01 232-281). In short, validation data serves model development; held-out test data serves evaluation; reports are evaluation regardless of which data they score.
- **R22.** Model/checkpoint saving, `save_pretrained`, checkpoint dicts, `load_model` / auto-resume for training → **model_generation** ("saving the trained model").
- **R23.** Training-history plots (`plt.plot(history['loss'])` after training) → **model_evaluation** (inspecting the trained model's performance). 
- **R24.** Metric helper functions (`def correct(...)`, `def metric_average(...)`) → model_evaluation, even when also called during training.
- **R25.** Functions and classes are labeled by their **content**, not their call site, and keep their label wherever they're defined (out-of-execution-order is fine): a single-stage `def train_epoch()` → model_generation; `class Net(nn.Module)` → model_generation; `def test()` → model_evaluation.
- **R26.** Cloud/platform data APIs (`Dataset.get_by_name`, `.download()`) → data_preparation. Session handles (`Run.get_context()`, workspace) follow data flow to their consumer (usually data_preparation).
- **R27.** Orchestrator calls that invoke a _single-stage_ function imported from elsewhere (`train_model(...)`, `load_and_prepare_data(...)`) → that stage, by data flow. Calls invoking _multi-stage_ functions → program_structure (R30).

---

## 6. program_structure

Definition: lines whose syntactic scope spans multiple stages. Exactly four cases:

- **R28.** The `def`/`return` lines of a multi-stage container function (`def train():` wrapping config→data→model→loop; `def main():`).
- **R29.** `if __name__ == "__main__":` and the bare call under it.
- **R30.** Top-level calls that invoke a multi-stage function (`main()`, `train()`).
- **R31.** `try`/`except`/`finally` wrapper lines around a multi-stage body derive like def headers: body uniform → the body's label

NOT program_structure: single-stage function defs (R25), argparse (R14 → env_config), orchestrator calls to single-stage functions (R27), and **loop/`with`/`if` headers**: these follow the normal anchor/data-flow rules like any code line (t2_07 line 38 → data_preparation via its `skf.split` anchor; t3_07 line 98 `with mlflow.start_run` → model_generation with its body's first block).

Motivation: semantic honesty; and it makes stage labels a complete routing table for task-file generation (env_config → dependency.py; stage blocks → task files; program_structure → discarded, replaced by generated run.py).

---

## 7. Taxonomy questions & rulings

- **ml_problem taxonomy** — allowed values: `classification-binary`, `classification-multiclass`, `regression`. Scope is **supervised only** currently.
- **Dual-task convention:** a script that trains both a classifier and a regressor (t3_01, t3_02) is labeled `regression` provisionally — the single-label `ml_problem` schema cannot express multi-task workflows; a known limitation.

---

## 8. Process discipline

- **R32.** Label from the source file and these rules only. **Never run the pipeline on a file (or look at its `results/*.json`) before its ground truth is written.**
- **R33.** Corpus files must be **author-published artifacts with a verifiable `source_url`** - the artifact studied must be the artifact its author published.
- **R34.** Metadata per file: `level` (1 = linear tutorial; 2 = structural complexity- functions/classes/hand-rolled loops/interleaving; 3 = real-world scale and noise- cloud APIs, tracking, notebook exports, delegation), `source_url`, `ml_problem`, `models`, `is_ml_training_workflow`. **Clause:** `models` descriptively lists models _present_ in the file, whether trained or loaded (t2_10's loaded Keras models count); extraction scoring, when built, targets positives only.
