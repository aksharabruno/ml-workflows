# main.py
import json
import sys
from pathlib import Path
from ast_parser import parse_file as parse_static
from llm_detector import run_llm_analysis


def _expand_llm_labels(stage_labels: dict) -> dict[int, list[str]]:
    """Expand LLM stage_labels (which may have ranges like '10-15') to per-line dict."""
    expanded: dict[int, list[str]] = {}
    for key, stages in stage_labels.items():
        parts = key.split("-")
        start = int(parts[0])
        end = int(parts[1]) if len(parts) > 1 else start
        for lineno in range(start, end + 1):
            expanded[lineno] = stages
    return expanded


def compare_results(ast_labels: dict, llm_labels: dict) -> dict:
    """Compare AST and LLM stage labels line by line."""
    ast_by_line: dict[int, list[str]] = {int(k): v for k, v in ast_labels.items()}
    llm_by_line = _expand_llm_labels(llm_labels)

    all_lines = sorted(set(ast_by_line) | set(llm_by_line))

    agree = {}
    conflict = {}
    ast_only = {}
    llm_only = {}

    for line in all_lines:
        ast_stages = set(ast_by_line.get(line, []))
        llm_stages = set(llm_by_line.get(line, []))
        key = str(line)

        if ast_stages and llm_stages:
            if ast_stages == llm_stages:
                agree[key] = sorted(ast_stages)
            else:
                conflict[key] = {
                    "ast": sorted(ast_stages),
                    "llm": sorted(llm_stages),
                }
        elif ast_stages:
            ast_only[key] = sorted(ast_stages)
        else:
            llm_only[key] = sorted(llm_stages)

    return {
        "agree": agree,
        "conflict": conflict,
        "ast_only": ast_only,
        "llm_only": llm_only,
    }


def run(input_path: Path):
    result_dir = Path(__file__).resolve().parent / "results"
    result_dir.mkdir(exist_ok=True)

    # Layer 1+2: AST static analysis + STAGE_MAP
    parser_result = parse_static(input_path)

    # Layer 3: LLM semantic analysis (pass source code so LLM can fill gaps)
    source_code = input_path.read_text()
    llm_result = run_llm_analysis(parser_result, source_code=source_code)

    # Layer 4: Compare AST and LLM outputs
    comparison = compare_results(
        parser_result.get("stage_labels", {}),
        llm_result.get("stage_labels", {}),
    )

    # Combine outputs
    final_result = {
        "file": input_path.name,
        "static_analysis": parser_result,
        "llm_analysis": llm_result,
        "comparison": comparison,
    }
    
    # Save final output
    result_path = result_dir / f"{input_path.stem}_result.json"
    with open(result_path, "w") as f:
        json.dump(final_result, f, indent=2)
    
    print(json.dumps(final_result, indent=2))
    print(f"\nSaved to {result_path}")
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_file>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)
    
    run(input_path)