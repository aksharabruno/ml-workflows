# Ground-Truth Labeling Rules

Annotator guide for `ground_truth_v2.json`. Every line of a script receives exactly one
stage label. These rules exist so that two annotators labeling the same file
independently produce the same result. Rules marked **[OPEN]** are pending ratification —
label provisionally and flag the file.

Taxonomy alignment: the four ML stages follow the AutoML phase consolidation of
Rajenthiram et al. 2025 (Table 2 settles scaling/encoding/augmentation boundaries);
the two code-level labels (`environment_configuration`, `program_structure`) are our
extensions, required for total line coverage of real code.

---

## 1. The labels

| Label | One-line definition |
|---|---|
| `environment_configuration` | Sets up the conditions for computation; no data, features, or model flow through it |
| `data_preparation` | Loading, cleaning, exploring (EDA), and transforming data into model-ready form |
| `feature_engineering` | Selecting, constructing, or extracting features to improve the input representation |
| `model_generation` | Defining the model, training it, tuning it, saving it |
| `model_evaluation` | Assessing the trained model on held-out data: predictions for scoring, metrics, reporting |
| `program_structure` **[OPEN]** | Lines whose syntactic scope spans multiple stages (wrapper defs, main guards) |

A file may omit any ML stage. A training script with no held-out evaluation has no
`model_evaluation` block (see R20). A file may also be a negative:
`is_ml_training_workflow: false` for inference-only or non-ML code.

---

## 2. Core decision procedure

Apply in order:

**Step 1 — Anchors.** Lines that directly perform a stage-defining action take that
stage's label: `read_csv` → data_prep, `PCA(...)` → feature_eng, `model.fit(...)` →
model_gen, `accuracy_score(...)` → model_eval, `import` → env_config.

**Step 2 — Data flow.** Every other line that computes something belongs to the stage
that **consumes its output**: a `params` dict used by `lgb.train` → model_generation;
a path variable used by `read_csv` → data_preparation; `num_labels = len(set(labels))`
used by the model constructor → model_generation. If output is consumed by multiple
stages, use the **first** consumer.

**Step 3 — Glue.** Lines that compute nothing (blanks, comments, bare prints):

- **R1.** Comments and status prints attach **forward** to the block they introduce
  (`# train`, `print("Loading data...")`).
- **R2.** Result-reporting prints attach **backward** to the stage that produced the
  value (`print(f"RMSE: {rmse}")` → model_evaluation). This is Step 2 applied to prints.
- **R3.** **Blank lines attach backward**: a blank line belongs to the same block as the
  nearest preceding non-blank line. A block never starts with a blank line.
- **R4.** Trailing glue at end of file attaches backward (there is no next block).

Glue placement never affects evaluation scores (blanks/comments/bare prints are masked)
— these rules exist for consistency and readability only.

---

## 3. Block mechanics

- **R5.** Blocks are contiguous and gap-free: every line 1..N belongs to exactly one block.
- **R6.** Blocks are per-BLOCK, not per-stage: a stage may appear multiple times
  (interleaving is normal at Levels 2–3). Never merge same-stage blocks across an
  intervening different-stage block — that would misrepresent execution order.
- **R7.** Single-line blocks are legal (`"73": ["environment_configuration"]` for a lone `hvd.init()`).
- **R8.** Do not split a logical unit (a loop body, a function body, a multi-line call)
  across two blocks unless it genuinely spans stages.
- **R9.** Ranges must not overshoot the file's line count.

---

## 4. environment_configuration — what counts

Litmus test: *would this line survive unchanged if you swapped in a different dataset
and model?* Yes → env_config.

- **R10.** Imports; runtime library config (`gdal.UseExceptions()`, `warnings.filterwarnings`).
- **R11.** Hardware/compute setup: device selection, `cuda.is_available()`, `set_device`.
  But moving a model/tensor to a device (`model.to(device)`) → the model/tensor's stage.
- **R12.** Distributed runtime init (`hvd.init()`) — but broadcasts of model/optimizer
  state → model_generation (state flows through them).
- **R13.** Random seeds, even mid-file.
- **R14.** CLI/config parsing: the whole argparse block including arg unpacking; Config
  class unpacking (`BATCH_SIZE = Config.BATCH_SIZE`). Rationale: args feed many stages;
  data-flow would shred the block.
- **R15.** Experiment-tracking *setup* (TensorBoard writer, logdir creation, MLflow init).
  *Using* the tracker (`run.log(x)`) → the stage that produced x.
- **R16.** Version asserts and compatibility checks.
- Plot *styling* (`sns.set`, `rcParams`) is NOT env_config → it serves the plotting
  stage (usually data_preparation EDA). Global runtime config → env; stage-serving
  config → that stage.

---

## 5. Boundary rulings (the case law)

- **R17.** Scaling/normalization/encoding — including of targets (`to_categorical(y)`,
  `LabelEncoder` on y) and format conversions (reshape, astype, `torch.tensor(x)`)
  → **data_preparation** (paper Table 2). NOT feature engineering.
- **R18.** **[OPEN — transforms/augmentation]** Image/tensor transform pipelines:
  format-required transforms (Resize, ToTensor, plain rescale /255) → data_preparation,
  by R17 analogy. Whether *augmentation* (random flips/crops, mixup) is data_preparation
  (paper says yes) or feature_engineering (passes the "affects quality" litmus test)
  is unratified. Current draft position: everything → data_preparation, per paper.
- **R19.** Feature engineering = selection (including manual column picking,
  `X = df[cols]`), construction, extraction (PCA, tokenization/vectorization,
  embedding-matrix construction). Text tokenization → feature_engineering.
- **R20.** In-loop validation (per-epoch val phase inside the training loop) →
  **model_generation** (training monitoring). Only held-out scoring after training is
  model_evaluation. Consequence: a script may legitimately have no model_evaluation
  block. Training-accuracy tracking inside the train phase is always model_generation.
  **[Ratify: this convention was contested during t2_06 labeling.]**
- **R21.** Model/checkpoint saving, `save_pretrained`, checkpoint dicts, `load_model` /
  auto-resume for training → **model_generation** ("saving the trained model").
- **R22.** Training-history plots (`plt.plot(history['loss'])` after training) →
  **model_evaluation** (inspecting the trained model's performance). **[Ratify:
  the LLM consistently prefers model_generation here.]**
- **R23.** Metric helper functions (`def correct(...)`, `def metric_average(...)`) →
  model_evaluation, even when also called during training. **[Ratify: weak precedent.]**
- **R24.** Functions and classes are labeled by their **content**, not their call site,
  and keep their label wherever they're defined (out-of-execution-order is fine):
  a single-stage `def train_epoch()` → model_generation; `class Net(nn.Module)` →
  model_generation; `def test()` → model_evaluation.
- **R25.** Cloud/platform data APIs (`Dataset.get_by_name`, `.download()`) →
  data_preparation. Session handles (`Run.get_context()`, workspace) follow data flow
  to their consumer (usually data_preparation). **[Ratify: could argue env_config.]**
- **R26.** Orchestrator calls that invoke a *single-stage* function imported from
  elsewhere (`train_model(...)`, `load_and_prepare_data(...)`) → that stage, by data
  flow. Calls invoking *multi-stage* functions → program_structure (R27).

---

## 6. program_structure **[OPEN — proposed 6th label]**

Definition: lines whose syntactic scope spans multiple stages. Exactly three cases:

- **R27.** The `def`/`return` lines of a multi-stage container function
  (`def train():` wrapping config→data→model→loop; `def main():`).
- **R28.** `if __name__ == "__main__":` and the bare call under it.
- **R29.** Top-level calls that invoke a multi-stage function (`main()`, `train()`).

NOT program_structure: single-stage function defs (R24), argparse (R14 → env_config),
orchestrator calls to single-stage functions (R26).

Motivation: semantic honesty; and it makes stage labels a complete routing table for
task-file generation (env_config → dependency.py; stage blocks → task files;
program_structure → discarded, replaced by generated run.py).

Fallback if not ratified: wrapper def lines attach forward to the first stage inside;
main guards attach backward to the last block (t1_01/t2_03 precedent).

---

## 7. Other open questions **[OPEN]**

- **model_application**: inference-heavy code that *applies* a trained model to new data
  at scale (t3_01's chunked raster prediction, report generation). No honest home in the
  current taxonomy — candidates: new label, or scope-limit the claim.
- **ml_problem taxonomy**: anomaly detection ruled "classification" (user decision,
  t2_03); clustering/other still untested.

---

## 8. Process discipline

- **R30.** Label from the source file and these rules only. **Never run the pipeline on
  a file (or look at its `results/*.json`) before its ground truth is written.** Files
  violating this must be flagged (currently: t3_01).
- **R31.** When a file forces a decision these rules don't cover: make the minimal
  call, mark it **[OPEN]** here, flag the file, and raise it for ratification. Never
  let two files embody two different answers to the same question.
- **R32.** Metadata per file: `level` (1 = linear tutorial; 2 = structural complexity —
  functions/classes/hand-rolled loops/interleaving; 3 = real-world scale and noise —
  cloud APIs, tracking, notebook exports, delegation), `source_url`, `ml_problem`,
  `models`, `is_ml_training_workflow`.
