import sys
import os
from llm_detector import run_llm_detection
from ml_parser import MLParser

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <python_file>")
        return
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File not found. Exiting.")
        return

    print("\n--- Running AST-based ML Detection ---")
    parser = MLParser()
    has_ml_workflow = parser.parse_file(file_path)

    if has_ml_workflow: #Only run LLM if AST detects ML workflow
        print("\n--- Running LLM-based ML Detection ---")
        run_llm_detection(file_path)


if __name__ == "__main__":
    main()