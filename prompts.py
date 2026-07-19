# Stage taxonomy aligned with Rajenthiram et al. 2025 ("AutoML: A Tertiary Study of
# Phases, Methods, Tools, and Frameworks"), extended with two code-level labels
# (environment_configuration, program_structure) for total line coverage.
# Conventions mirror LABELING_RULES.md as ratified 2026-07-11. Freeze candidate —
# not yet frozen. Freeze happens after t3 GT review, then no edits without
# re-running the full multi-run benchmark.
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

TASK_DEFINITIONS = """
## 1. Environment Configuration
  - library_loading
      The process of importing and configuring required software libraries and packages needed
      for the data science project.
      Examples: `import pandas as pd`, `from sklearn.ensemble import RandomForestClassifier`

  - data_loading
      The process of acquiring and reading datasets from local or remote sources into the
      computational environment. Only applies to EXTERNAL sources — files, databases, URLs,
      or APIs. Synthetic data generated with np.random or hardcoded arrays is NOT data_loading.
      Examples: `pd.read_csv('data.csv')`, `fetch_20newsgroups()`, `open('file.txt')`

## 2. Data Preparation and Exploration
  - data_preparation
      The process of structuring and organizing raw data into a suitable format for analysis,
      including handling missing values, duplicates, and data type conversions.
      Examples: `df.fillna(0)`, `df.astype(int)`, `torch.tensor(x)`, `DataLoader(...)`

  - exploratory_data_analysis
      The process of examining and visualizing data to assess quality, understand distributions,
      detect noise, skewness, correlations, and class imbalances. Data is NOT modified.
      Examples: `df.describe()`, `sns.heatmap(df.corr())`, `plt.show()`, `df.value_counts()`

  - data_cleaning
      The process of detecting and correcting errors, noise, and inconsistencies in the data.
      Examples: `df.dropna()`, `df.drop_duplicates()`, `df['col'].replace(' ?', np.NaN)`

## 3. Feature Engineering and Selection
  - feature_engineering
      The process of applying transformation and encoding functions to existing variables to
      create more informative representations for modeling.
      Examples: `pd.get_dummies(df)`, `LabelEncoder().fit_transform(y)`, `tokenizer.encode(text)`

  - feature_transformation
      The process of applying mathematical or statistical operations to alter the feature space
      (e.g. normalization, dimensionality reduction) to improve modeling accuracy.
      Examples: `StandardScaler().fit_transform(X)`, `PCA(n_components=10)`, `np.log1p(x)`

  - feature_selection
      The process of identifying and retaining the most relevant subset of features while
      removing unnecessary or redundant ones.
      Examples: `SelectKBest(k=10)`, `df[['col1', 'col2']]`, `RFE(estimator=...)`

## 4. Model Building and Selection
  - model_building
      The process of selecting and instantiating the model algorithm or neural architecture
      suited for the problem — not yet fitting to data.
      Examples: `RandomForestClassifier(n_estimators=100)`, `nn.Linear(128, 10)`, `model.compile(...)`

  - train_test_splitting
      The process of dividing data into training, validation, and test subsets.
      Examples: `train_test_split(X, y, test_size=0.2)`, `StratifiedKFold(n_splits=5)`

  - model_training
      The process of fitting the model to training data to learn patterns. Includes both
      high-level API calls and hand-rolled loops with backward passes.
      Examples: `model.fit(X_train, y_train)`, `loss.backward(); optimizer.step()`

  - model_parameter_tuning
      The process of optimizing hyperparameters using search or optimization methods.
      Examples: `GridSearchCV(...)`, `RandomizedSearchCV(...)`, `optuna.create_study()`

  - model_validation
      The process of evaluating model performance using metrics, or combining models into
      an ensemble to improve predictive accuracy.
      Examples: `model.predict(X_test)`, `accuracy_score(y_test, y_pred)`, `model.evaluate(...)`
"""


def build_stage_labeling_prompt(source_code: str) -> str:
    numbered_source = "\n".join(
        f"{i}: {line}" for i, line in enumerate(source_code.splitlines(), start=1)
    )
    return f"""
You are an ML pipeline stage classifier. Given a Python script, divide it into contiguous stage blocks and label each block with one of the 6 stages below.

## The 6 valid stage labels (use ONLY these exact strings):
- "environment_configuration"
- "data_preparation"
- "feature_engineering"
- "model_generation"
- "model_evaluation"
- "program_structure"

## Stage definitions:
{STAGE_DEFINITIONS}

## Source code (each line is prefixed with its line number as "N:"):
```python
{numbered_source}
```

## Your task:
1. Divide the script into contiguous blocks — every line must belong to exactly one block, no gaps.
2. Label each block with one of the 6 stages above.
3. Identify the ML problem type: "classification", "regression", "clustering", or "other".

Use the "N:" line-number prefixes for all start/end values — do not count lines yourself. The blocks must cover line 1 through the last numbered line exactly.

Rules:
- Every line must be covered including comments, blank lines, print statements, and variable assignments.
- Data-flow rule: a computed line belongs to the stage that CONSUMES its output (a params dict used by a fit call → model_generation; a path variable used by read_csv → data_preparation; `num_labels = len(set(labels))` used by the model constructor → model_generation). If the output is consumed by multiple stages, use the FIRST consumer. Exception: whole constants/argparse/Config blocks stay together as environment_configuration (see definition 1).
- Glue lines: comments and status prints (`# train`, `print("Loading data...")`) attach FORWARD to the block they introduce; result-reporting prints (`print(f"RMSE: {{rmse}}")`) attach BACKWARD to the stage that produced the value; blank lines attach BACKWARD to the nearest preceding non-blank line — a stage block never starts with a blank line.
- Import lines always belong to "environment_configuration", even mid-file.
- Blocks must be contiguous and non-overlapping. The same stage may appear as multiple blocks (interleaving is normal); never merge two same-stage blocks across an intervening different-stage block.
- The model_generation / model_evaluation boundary is where TRAINING ENDS: `.fit(...)` and everything that steers training (per-epoch validation inside the loop, early stopping, checkpointing) is model_generation. Post-training `.predict`/`.evaluate`/metric calls on held-out data, training-history plots, and metric helper defs are model_evaluation. A metric call INSIDE the training loop is model_generation; the same call after the loop is model_evaluation. Two post-training refinements: decision-threshold selection on VALIDATION data (searching a cutoff, then using it) is tuning — model_generation; diagnostics REPORTED on training data (OOB score, feature importances, training-set confusion) are model_evaluation.
- Functions and classes are labeled by their CONTENT, not their call site (a `class Net(nn.Module)` or `def train_epoch()` defined before the data is loaded is still model_generation). Only functions whose body spans multiple stages get "program_structure" on their def/return lines.
- Do NOT split a logical unit (a for-loop body, a function body, a multi-line call) across two blocks unless it genuinely spans stages.
- Use your understanding of the full script to make labeling decisions, not just individual lines in isolation.

Return ONLY a JSON object in this exact format — no markdown, no explanation:
{{
    "is_ml_training_workflow": true,
    "ml_problem": "classification",
    "stages": [
        {{"stage": "environment_configuration", "start": 1, "end": 5}},
        {{"stage": "data_preparation", "start": 6, "end": 15}},
        {{"stage": "model_generation", "start": 16, "end": 25}},
        {{"stage": "model_evaluation", "start": 26, "end": 29}},
        {{"stage": "program_structure", "start": 30, "end": 31}}
    ],
    "reasoning": "2-3 sentence summary of the workflow"
}}
"""
