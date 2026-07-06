import json
import sys
from pathlib import Path
from llm_detector import run_llm_analysis
from ast_parser import extract_from_stages


def run(input_path: Path):
    result_dir = Path(__file__).resolve().parent / "results"
    result_dir.mkdir(exist_ok=True)

    source_code = input_path.read_text(encoding="utf-8")

    # Layer 1: LLM detects stages and ml_problem
    llm_result = run_llm_analysis(source_code)  #returns a JSON object with stage labels
    if not llm_result:
        print("LLM analysis failed.")
        sys.exit(1)

    stages = llm_result.get("stages", [])

    # Layer 2: AST extracts hyperparameters, dataset, model from LLM regions
    ast_result = extract_from_stages(source_code, stages)

    final_result = {
        "file": input_path.name,
        "is_ml_training_workflow": llm_result.get("is_ml_training_workflow", False),
        "ml_problem": llm_result.get("ml_problem", "unknown"),
        "dataset": ast_result.get("dataset"),
        "model": ast_result.get("model"),
        "hyperparameters": ast_result.get("hyperparameters", []),
        "stages": stages,
        "reasoning": llm_result.get("reasoning", ""),
    }

    result_path = result_dir / f"{input_path.stem}_result.json"
    with open(result_path, "w") as f:
        json.dump(final_result, f, indent=2)

    print(json.dumps(final_result, indent=2))
    print(f"\nSaved to {result_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <path_to_file>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    run(input_path)
