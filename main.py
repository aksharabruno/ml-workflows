import csv
import os
from llm_detector import run_llm_detection, _parse_json_response
from ml_parser import MLParser

parser_out_path = 'results/ml_parser_results.csv'
llm_out_path = 'results/llm_detector_results.csv'

def append_to_csv(file_path, row, fieldnames):
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header only once
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def analyze_file(file_path):
    print(f"\n=== Analyzing: {file_path} ===")

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    print("\n--- Running AST-based ML Detection ---")
    parser = MLParser()
    ml_parser_output = parser.parse_file(file_path)

    print("\n--- Running LLM-based ML Detection ---")
    llm_detector_output = run_llm_detection(file_path)


    # ===== Write to CSV =====
    fieldnames = ['filename', 'is_ml_workflow', 'data_loading', 'training', 'evaluation']

    append_to_csv(parser_out_path, ml_parser_output, fieldnames)

    append_to_csv(llm_out_path, llm_detector_output, fieldnames)



def analyze_all(data_dir):
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return

    files = sorted(os.listdir(data_dir))
    for fname in files:
        if not fname.endswith('.py'):
            continue
        analyze_file(os.path.join(data_dir, fname))


def main():
    # Interactive mode
    print("Select analysis mode:")
    print("1. One file at a time")
    print("2. All files in dataset (data/)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == '1':
        fname = input("Enter filename (path or name inside data/): ").strip()
        if os.path.exists(fname):
            analyze_file(fname)
        else:
            data_path = os.path.join('data', fname)
            if os.path.exists(data_path):
                analyze_file(data_path)
            else:
                print(f"Error: File not found: {fname}")

    elif choice.startswith('2'):
        analyze_all('data')

    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()