# main.py
import json
import sys
from pathlib import Path
from headergen_parser import generate_headergen_annotations, parse_headergen_output
from llm_detector import run_llm_analysis

def run(input_path: Path):
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Layer 1: HeaderGen
    generate_headergen_annotations(input_path, output_dir)
    
    json_path = output_dir / f"{input_path.stem}.json"
    if not json_path.exists():
        print(f"No HeaderGen output found for {input_path.name}")
        return
    
    # Layer 2: Static parser + STAGE_MAP
    parser_result = parse_headergen_output(json_path)
    
    # Layer 3: LLM semantic analysis
    llm_result = run_llm_analysis(parser_result)
    
    # Combine outputs
    final_result = {
        "file": input_path.name,
        "static_analysis": parser_result,
        "llm_analysis": llm_result,
    }
    
    # Save final output
    result_path = output_dir / f"{input_path.stem}_result.json"
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