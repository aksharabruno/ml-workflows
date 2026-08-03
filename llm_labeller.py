"""LLM labeling layer for the chunk pipeline.

Owns everything about talking to the model: client construction, model + token
config, the API call, and parsing the response into per-chunk stage labels. The
pipeline calls label_chunks() and never touches the anthropic SDK directly.
"""

import ast
import json
import re
from typing import Dict, Optional, Tuple

import anthropic
from dotenv import load_dotenv

from prompts import build_chunk_prompt

load_dotenv()

MODEL = "claude-opus-4-8"
MAX_TOKENS = 8192

VALID = {
    "environment_configuration", "data_preparation", "feature_engineering",
    "model_generation", "model_evaluation", "program_structure",
}


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


def label_chunks(source: str, chunks, quiet: bool = False) -> Tuple[Dict, Dict[int, str]]:
    """Send the chunked script to the LLM and return (parsed, llm_labels).

    parsed      : full response dict (is_ml_training_workflow, ml_problem, chunk_labels).
    llm_labels  : {chunk_id: stage} filtered to the VALID stage set.
    """
    prompt = build_chunk_prompt(source, chunks)
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text_block = next((b for b in response.content if b.type == "text"), None)
    if text_block is None:
        raise RuntimeError("no text block in LLM response")
    parsed = _parse_json_response(text_block.text, verbose=not quiet)
    if not parsed:
        raise RuntimeError("could not parse LLM response")

    raw = parsed.get("chunk_labels", {})
    llm_labels = {int(k): v for k, v in raw.items()
                  if isinstance(v, str) and v in VALID}
    return parsed, llm_labels
