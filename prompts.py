# Stage taxonomy aligned with Rajenthiram et al. 2025 ("AutoML: A Tertiary Study of
# Phases, Methods, Tools, and Frameworks"), extended with environment_configuration
# for code-level labeling. Scaling/normalization/encoding belong to data_preparation
# (their Table 2); feature_engineering covers selection/construction/extraction only.
STAGE_DEFINITIONS = """
## 1. Environment Configuration
   Importing and configuring all required libraries and packages.
   Examples: `import pandas as pd`, `from sklearn.ensemble import RandomForestClassifier`, `import torch`

## 2. Data Preparation
   Loading, cleaning, exploring, and transforming raw data into a form suitable for modelling.
   Includes: reading files/databases, handling missing values, EDA plots and profiling,
   scaling/normalization, encoding (including target labels, e.g. one-hot), data type
   conversions, train/test splitting, and DataLoader setup.
   Examples: `pd.read_csv(...)`, `df.fillna(0)`, `StandardScaler().fit_transform(X)`,
             `keras.utils.to_categorical(y)`, `train_test_split(...)`, `DataLoader(...)`

## 3. Feature Engineering
   Selecting, constructing, or extracting features to improve the model's input representation.
   Includes: feature selection (choosing a subset), feature construction (creating new
   variables), and feature extraction (dimensionality reduction, domain-specific encodings).
   Feature selection includes manually picking a column subset, not just algorithmic selection.
   NOT included: scaling/normalization or label encoding — those are data preparation.
   Examples: `SelectKBest(k=10)`, `PCA(n_components=10)`, `pd.get_dummies(df)`,
             `TfidfVectorizer().fit_transform(corpus)`, `X = df[['col1', 'col2']]`

## 4. Model Generation
   Defining the model and fitting it to training data, including hyperparameter tuning.
   Includes: model/architecture instantiation, compile/fit calls, hand-rolled training
   loops, hyperparameter search, and saving the trained model.
   Examples: `RandomForestClassifier(n_estimators=100)`, `keras.Sequential([...])`,
             `model.fit(X_train, y_train)`, `loss.backward(); optimizer.step()`,
             `GridSearchCV(...)`

## 5. Model Evaluation
   Assessing the trained model's performance on held-out data.
   Includes: predictions made for scoring, computing metrics, validation, and reporting results.
   Examples: `model.predict(X_test)`, `accuracy_score(y_test, y_pred)`,
             `mean_squared_error(y_test, y_pred)`, `model.evaluate(...)`
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
You are an ML pipeline stage classifier. Given a Python script, divide it into contiguous stage blocks and label each block with one of the 5 stages below.

## The 5 valid stage labels (use ONLY these exact strings):
- "environment_configuration"
- "data_preparation"
- "feature_engineering"
- "model_generation"
- "model_evaluation"

## Stage definitions:
{STAGE_DEFINITIONS}

## Source code (each line is prefixed with its line number as "N:"):
```python
{numbered_source}
```

## Your task:
1. Divide the script into contiguous blocks — every line must belong to exactly one block, no gaps.
2. Label each block with one of the 5 stages above.
3. Identify the ML problem type: "classification", "regression", "clustering", or "other".

Use the "N:" line-number prefixes for all start/end values — do not count lines yourself. The blocks must cover line 1 through the last numbered line exactly.

Rules:
- Every line must be covered including comments, blank lines, print statements, and variable assignments. These are glue lines — assign them to whichever stage they contextually support: a line belongs to the stage that consumes its output (e.g. a params dict used by a fit call is model_generation), and comments/prints attach to the block they introduce.
- Blank lines belong to the same block as the nearest preceding non-blank line (i.e. they close the block above; a stage block never starts with a blank line).
- Import lines always belong to "environment_configuration".
- Blocks must be contiguous and non-overlapping.
- The boundary between model_generation and model_evaluation is where training/saving ends and scoring begins. Every `.predict(...)`, `.evaluate(...)`, or metric call (accuracy_score, mean_squared_error, ...) on held-out/test data belongs to model_evaluation, never model_generation. The `.fit(...)` call itself is always model_generation, even when scoring follows immediately.
- Do NOT split a logical unit (e.g. a for-loop or function definition) across two blocks unless it genuinely spans two stages.
- Use your understanding of the full script to make labeling decisions, not just individual lines in isolation.

Return ONLY a JSON object in this exact format — no markdown, no explanation:
{{
    "is_ml_training_workflow": true,
    "ml_problem": "classification",
    "stages": [
        {{"stage": "environment_configuration", "start": 1, "end": 5}},
        {{"stage": "data_preparation", "start": 6, "end": 15}},
        {{"stage": "model_generation", "start": 16, "end": 25}},
        {{"stage": "model_evaluation", "start": 26, "end": 30}}
    ],
    "reasoning": "2-3 sentence summary of the workflow"
}}
"""
