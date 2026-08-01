# Stage taxonomy aligned with Rajenthiram et al. 2025 ("AutoML: A Tertiary Study of
# Phases, Methods, Tools, and Frameworks"), extended with two code-level labels
# (environment_configuration, program_structure) for total line coverage.
STAGE_DEFINITIONS = """
## 1. Environment Configuration
   Sets up the conditions for computation; no data, features, or model flow through it.
   Includes: imports (ALWAYS env, even mid-file), runtime library config
   (`warnings.filterwarnings`, `gdal.UseExceptions()`), device selection
   (`torch.device(...)`, `cuda.is_available()`), random seeds (even mid-file),
   version asserts, CLI/argparse blocks including argument unpacking, Config-class
   unpacking, bare constants/config blocks (`BATCH_SIZE = 32`, `EPOCHS = 10`, paths)
   WHOSE VALUES FEED SEVERAL DIFFERENT STAGES — keep such a block together, do NOT
   split it by usage. But a dict or constants group consumed by a SINGLE call
   (a `params = {...}` dict passed to `lgb.train`/`model.fit`) is NOT env_config —
   it follows data flow to its consumer (usually model_generation) — and
   experiment-tracking SETUP (TensorBoard/MLflow/wandb init, logdir creation),
   and loading pretrained PROCESSING artifacts — tokenizers/vectorizers via
   `from_pretrained` or hub fetch (loading a pretrained MODEL, by contrast,
   is model_generation).
   NOT env: plot styling (`sns.set`, `rcParams`) — that belongs to the stage whose
   plot it styles; moving a model/tensor to a device (`model.to(device)`) — that
   belongs to the model/tensor's stage.
   Examples: `import torch`, `torch.manual_seed(42)`, `parser.add_argument(...)`,
             `BATCH_SIZE = Config.BATCH_SIZE`, `LR = 3e-4`, `mlflow.set_experiment(...)`

## 2. Data Preparation
   Loading, cleaning, exploring, and transforming data into model-ready form.
   Includes: reading files/databases/cloud datasets, synthetic data generation,
   handling missing values (`np.nan_to_num`, `df.fillna`), EDA plots and profiling,
   scaling/normalization, encoding of features AND targets (`to_categorical(y)`,
   `LabelEncoder` on y), format conversions (reshape, astype, `torch.tensor(x)`,
   `lgb.Dataset(...)`), image/tensor transform pipelines INCLUDING augmentation
   (Resize, ToTensor, Normalize, RandomFlip/Rotation/ColorJitter — ALL data_preparation),
   train/test/validation splitting (incl. K-fold splits), Dataset wrappers and
   DataLoader setup.
   Examples: `pd.read_csv(...)`, `transforms.Compose([...])`, `train_test_split(...)`,
             `StandardScaler().fit_transform(X)`, `DataLoader(...)`, `random_split(...)`

## 3. Feature Engineering
   Selecting, constructing, or extracting features to improve the input representation.
   Includes: feature selection (incl. manually picking a column subset), feature
   construction, extraction (PCA, dimensionality reduction), text tokenization /
   vectorization, and embedding-matrix construction.
   NOT included: scaling/normalization, label encoding, or transform pipelines —
   those are data preparation.
   Examples: `SelectKBest(k=10)`, `PCA(n_components=10)`, `tokenizer(texts, ...)`,
             `TfidfVectorizer().fit_transform(corpus)`, `X = df[['col1', 'col2']]`

## 4. Model Generation
   Defining the model, training it, tuning it, saving/restoring it.
   Includes: model/architecture instantiation (incl. custom nn.Module classes),
   compile/fit calls, hand-rolled training loops, optimizers/schedulers/criteria,
   hyperparameter search, decision logic that STEERS training (early stopping,
   best-model tracking), checkpoint saving AND loading/auto-resume
   (`torch.save`, `save_pretrained`, `load_model` before/for training),
   and PER-EPOCH VALIDATION INSIDE THE TRAINING LOOP — computing val loss/accuracy
   each epoch is training monitoring, NOT model_evaluation. Training-accuracy
   tracking inside the loop is likewise model_generation.
   Examples: `keras.Sequential([...])`, `model.fit(...)`, `loss.backward()`,
             `GridSearchCV(...)`, `torch.save(checkpoint, path)`,
             `val_ret = test(validation_loader, ...)  # inside epoch loop`

## 5. Model Evaluation
   Assessing the trained model AFTER training completes: predictions for scoring,
   metrics on held-out data, and reporting.
   Includes: post-training `.predict`/`.evaluate` on test data, metric calls
   (`accuracy_score`, `f1_score`, `classification_report`, `confusion_matrix`),
   metric HELPER function defs (`def correct(...)`, `def compute_metrics(...)`) even
   when also called during training, training-history plots after training
   (`plt.plot(history['loss'])` — model_evaluation, NOT model_generation),
   confusion-matrix/ROC plots, and result reporting/export.
   A script whose only validation happens inside the training loop has NO
   model_evaluation block — that is normal.
   Examples: `model.evaluate(X_test, y_test)`, `accuracy_score(y_test, y_pred)`,
             `plt.plot(history['val_loss'])`, `classification_report(...)`

## 6. Program Structure
   Lines whose syntactic scope spans multiple stages. EXACTLY these cases:
   (a) the def/return lines of a function whose body spans multiple stages
       (`def main():`, `def train():` wrapping config→data→model→eval),
   (b) `if __name__ == "__main__":` and the bare call under it,
   (c) top-level calls that invoke a multi-stage function (`main()`, `train()`).
   NOT program_structure: single-stage function defs (a pure `def train_epoch()` is
   model_generation; a pure `def test()` is model_evaluation — label functions by
   their CONTENT), argparse blocks (env), calls to single-stage functions (label
   by the called function's stage), and try/except or with blocks whose body is a
   single stage (a try around tracking setup is environment_configuration).
   Examples: `def main():`, `if __name__ == "__main__":`, `    main()`
"""

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
"ml_problem" must be one of: "classification-binary", "classification-multiclass", "regression", "clustering", "other".
Return ONLY a JSON object, no markdown, no explanation:
{{
    "is_ml_training_workflow": true,
    "ml_problem": "classification",
    "chunk_labels": {{ {", ".join(f'"{i}": "<stage>"' for i in need[:3])}, ... }}
}}
"chunk_labels" must contain every chunk id marked LABEL THIS: {", ".join(need)}
"""