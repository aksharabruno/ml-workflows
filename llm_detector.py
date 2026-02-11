import re
import ollama
import json
from typing import Dict, Optional

def run_llm_detection(file_path):
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

    response = ollama.generate(model='llama3', prompt=prompt)
    raw_response = response['response']
    _print_results(_parse_json_response(raw_response))


def _parse_json_response(raw_response: str) -> Optional[Dict]:
    """
    Extract and parse JSON from LLM response
    Handles cases where LLM adds text before/after JSON
    """
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        pass

    json_start = raw_response.find('{')
    json_end = raw_response.rfind('}') + 1
    
    if json_start != -1 and json_end > json_start:
        json_str = raw_response[json_start:json_end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass
    
    return None
    
def _print_results(result: Dict):
    print(f"\nML Workflow: {result.get('is_ml_workflow', False)}")
    print(f"Reasoning: {result.get('reasoning', 'N/A')}")
    
    stages = result.get('stages', {})
    print(f"\nDetected Stages:")
    print(f"  Data Loading (lines): {stages.get('data_loading', [])}")
    print(f"  Training (lines): {stages.get('training', [])}")
    print(f"  Evaluation (lines): {stages.get('evaluation', [])}")
