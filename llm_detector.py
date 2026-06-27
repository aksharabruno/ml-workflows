import re
import os
import anthropic
import json
import ast
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

STAGE_DEFINITIONS = """
    1.  library_loading        – Any import or from-import statement. No computation occurs.
    2.  data_loading           – Reading data into memory from a file, database, or URL (e.g. pd.read_csv, fetch_20newsgroups).
    3.  data_preparation       – Cleaning, reshaping, or converting data into a usable form (e.g. tensor conversion, DataLoader setup, padding).
    4.  exploratory_data_analysis – Visualising or summarising data without modifying it (e.g. plt.show, df.describe, seaborn plots).
    5.  data_cleaning          – Removing or correcting invalid/missing data (e.g. dropna, fillna, replace).
    6.  feature_engineering    – Creating new features from existing ones (e.g. arithmetic combinations, date extraction).
    7.  feature_transformation – Scaling, encoding, or normalising existing features (e.g. StandardScaler, tokenizer.encode, fit_transform).
    8.  feature_selection      – Selecting a subset of columns or features (e.g. df[['f1','f2']], SelectKBest).
    9.  model_building         – Instantiating a model, defining architecture, or setting hyperparameters — not yet fitting (e.g. RandomForestClassifier(), nn.Linear).
    10. train_test_splitting   – Partitioning data into train/test/val sets (e.g. train_test_split).
    11. model_training         – Fitting the model to training data (e.g. model.fit, optimizer.step in a training loop).
    12. model_parameter_tuning – Searching over hyperparameter configurations (e.g. GridSearchCV, optuna).
    13. model_validation       – Generating predictions and computing metrics (e.g. model.predict, accuracy_score, model.evaluate).
"""


def run_llm_analysis(parser_output, source_code: str = ""):
    already_detected = parser_output.get("stages_detected", {})
    covered_lines = list(already_detected.keys())

    prompt = f"""
        You are an ML pipeline stage classifier. Your job is to assign stage labels to lines of a Python script that a static analysis tool could not label.

        ## The 13 valid stage labels (use ONLY these exact strings):
        {STAGE_DEFINITIONS}

        ## What static analysis already detected:
        The following stages were already identified by a STAGE_MAP (keyword-to-stage lookup):
        {json.dumps(already_detected, indent=2)}

        ## Source code:
        ```python
        {source_code}
        ```

        ## Your task:
        Identify line ranges that the static analysis MISSED — specifically:
        - User-defined functions or classes that wrap ML operations (e.g. a `train()` function containing a training loop)
        - Hand-rolled loops performing model training or validation
        - Any other ML-relevant lines not covered by the static analysis

        For each missed line range, assign one or more stage labels from the 13 above.
        Do NOT re-label lines already covered by static analysis.
        Do NOT invent new stage names.
        Skip lines with no ML relevance (utility code, logging, config parsing).

        Return ONLY a JSON object in this exact format:
        {{
            "stage_labels": {{
                "<start_line>-<end_line>": ["<stage_label>"],
                "<single_line>": ["<stage_label>", "<stage_label>"]
            }},
            "reasoning": "brief explanation of what the static layer missed and why"
        }}

        If nothing was missed, return: {{"stage_labels": {{}}, "reasoning": "Static analysis covered all relevant stages."}}
    """

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )
    text_block = next((b for b in response.content if b.type == "text"), None)
    if text_block is None:
        print("[LLM] No text block found in response")
        return None
    return _parse_json_response(text_block.text, verbose=True)




def _parse_json_response(raw_response: str, verbose: bool = False) -> Optional[Dict]:
    """
    Attempt to extract and parse a JSON object from an LLM response.

    Strategies:
    1. Direct JSON parsing
    2. Extract JSON from markdown code blocks
    3. Extract balanced JSON objects from text
    4. Fallback to Python literal_eval for non-strict JSON

    Returns:
        dict if successful, otherwise None
    """

    def log(msg):
        if verbose:
            print(f"[JSON PARSER] {msg}")

    # ---------- 1. Direct parse ----------
    try:
        return json.loads(raw_response)
    except Exception as e:
        log(f"Direct parse failed: {e}")

    # ---------- 2. Extract from code blocks ----------
    code_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", raw_response, re.DOTALL)

    for block in code_blocks:
        try:
            return json.loads(block)
        except Exception as e:
            log(f"Code block JSON parse failed: {e}")

    # ---------- 3. Extract balanced JSON objects ----------
    def extract_balanced_json(text: str):
        stack = []
        start = None

        for i, ch in enumerate(text):
            if ch == '{':
                if not stack:
                    start = i
                stack.append(ch)
            elif ch == '}':
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        yield text[start:i + 1]
                        start = None

    for candidate in extract_balanced_json(raw_response):
        try:
            return json.loads(candidate)
        except Exception as e:
            log(f"Balanced JSON parse failed: {e}")

    # ---------- 4. Fallback: Python-style dict ----------
    for candidate in extract_balanced_json(raw_response):
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            log(f"literal_eval failed: {e}")

    # ---------- 5. Give up ----------
    log("All parsing strategies failed")
    return None
    
def _print_results(result: Dict):   # may not be required?
    print(f"\nML Workflow: {result.get('is_ml_workflow', False)}")
    print(f"Reasoning: {result.get('reasoning', 'N/A')}")
    
    stages = result.get('stages', {})
    print(f"\nDetected Stages:")
    print(f"  Data Loading (lines): {stages.get('data_loading', [])}")
    print(f"  Training (lines): {stages.get('training', [])}")
    print(f"  Evaluation (lines): {stages.get('evaluation', [])}")
