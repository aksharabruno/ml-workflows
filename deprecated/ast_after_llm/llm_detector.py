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




