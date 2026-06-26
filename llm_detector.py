import re
import os
import anthropic
import json
import ast
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

def run_llm_analysis(parser_output):

    # may need to fine tune the prompt later
    prompt = f"""
        You are an ML optimization expert. Given this structured extraction from an ML script:

        {json.dumps(parser_output, indent=2)}

        Answer two questions:
        1. Do the detected stages correctly represent an ML training workflow?
        2. Which hyperparameters are optimisation targets (would meaningfully affect model performance if varied)?

        Return ONLY a JSON object:
        {{
            "stages_valid": boolean,
            "optimisation_targets": [
                {{"name": "param_name", "value": value, "tunable": boolean, "reason": "short reason"}}
            ],
            "reasoning": "short explanation"
        }}
    """
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_json_response(response.content[-1].text, verbose=True)




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
