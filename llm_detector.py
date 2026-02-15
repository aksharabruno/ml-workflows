import re
import os
import ollama
import json
import ast
from typing import Dict, Optional

def run_llm_detection(file_path):
    result = {
        'filename': os.path.basename(file_path),
        'is_ml_workflow': False,
        'data_loading': '[]',
        'training': '[]',
        'evaluation': '[]'
    }
    
    with open(file_path, 'r') as f:
        code_content = f.read()

    # may need to fine tune the prompt later
    prompt = f"""
    You are an expert MLOps analyzer. Analyze the following Python code.
    1. Determine if it is a Machine Learning training workflow.
    2. Identify the line numbers for: 'Data Loading', 'Model Training', and 'Evaluation'.
    
    Return the result ONLY as a JSON object with this structure:
    {{
        "is_ml_workflow": boolean,
        "stages": {{
            "data_loading": [start_line, end_line],
            "training": [start_line, end_line],
            "evaluation": [start_line, end_line]
        }},
        "reasoning": "short explanation"
    }}

    CODE:
    {code_content}
    """

    response = ollama.generate(model='llama3', prompt=prompt)   # definitely need a better model; llama is giving different values for same script in different iterations
    parsed_response = _parse_json_response(response['response'], verbose=True)  # parse the raw response

    if parsed_response:
        result['is_ml_workflow'] = parsed_response.get('is_ml_workflow', False)
        stages = parsed_response.get('stages', {})
        result['data_loading'] = json.dumps(stages.get('data_loading', []))
        result['training'] = json.dumps(stages.get('training', []))
        result['evaluation'] = json.dumps(stages.get('evaluation', []))

    print(f"\nLLM Detection Output:\n{json.dumps(result, indent=2)}")

    return result


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
