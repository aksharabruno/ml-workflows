STAGE_DEFINITIONS = """
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


def build_stage_labeling_prompt(already_detected: dict, source_code: str) -> str:
    import json
    return f"""
        You are an ML pipeline stage classifier. Given a Python script, label every ML-relevant line range with one or more stage labels from the list below.

        ## The 13 valid stage labels (use ONLY these exact strings):
        {STAGE_DEFINITIONS}

        ## Hints from static analysis:
        A keyword-based STAGE_MAP already detected the following stages. Use these as a starting reference, but you are NOT limited to them — label the full file independently and correct any mistakes you see.
        {json.dumps(already_detected, indent=2)}

        ## Source code:
        ```python
        {source_code}
        ```

        ## Your task:
        Produce a complete stage labeling of the entire file. For every ML-relevant line range, assign one or more labels from the 13 stages above.

        Rules:
        - Always label import/from-import lines as "library_loading".
        - Group consecutive lines that belong to the same stage into a single range (e.g. "1-5").
        - A line range may have multiple labels if it genuinely spans two stages (e.g. a single call that both builds and trains).
        - Skip lines with no ML relevance: comments, blank lines, print statements, logging, argparse config.
        - "data_loading" only applies to reading from an EXTERNAL source (file, database, URL, API). Synthetic data from np.random or hardcoded arrays is NOT data_loading.
        - "model_training" applies to hand-rolled loops with backward passes, not just model.fit calls.
        - Do NOT invent stage names outside the 13 defined above.

        Return ONLY a JSON object in this exact format:
        {{
            "stage_labels": {{
                "<start_line>-<end_line>": ["<stage_label>"],
                "<single_line>": ["<stage_label>", "<stage_label>"]
            }},
            "is_ml_training_workflow": true,
            "reasoning": "2-3 sentence summary of the workflow and any corrections to static analysis"
        }}
    """
