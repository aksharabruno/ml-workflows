import re
import os
import anthropic
import json
import ast
from typing import Dict, Optional
from dotenv import load_dotenv
from prompts import build_stage_labeling_prompt

load_dotenv()


def run_llm_analysis(source_code: str) -> Optional[Dict]:
    prompt = build_stage_labeling_prompt(source_code)

    source_lines = source_code.count("\n")
    max_tokens = max(4096, source_lines * 30)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=max_tokens,
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
    
