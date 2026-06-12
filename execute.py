import subprocess
from itertools import chain
from pathlib import Path
from headergen_parser import parse_headergen_output, generate_headergen_annotations

root = Path(__file__).resolve().parent

input_dir = root / "input"
output_dir = root / "output"

output_dir.mkdir(exist_ok=True)

# runs for jupyter notebooks and python scripts
files = chain(
    input_dir.glob("*.py"), input_dir.glob("*.ipynb")
)  

for input_file in files:
    print(f"Processing {input_file.name}")
    generate_headergen_annotations(input_file, output_dir)
    parser_output = parse_headergen_output(output_dir / f"{input_file.stem}.json")

    print("Is ML Training Workflow:", parser_output["is_ml_training_workflow"])  # for debugging; can remove later
    print("Stages Detected:", list(parser_output["stages_detected"].keys()))
    print("Models:", parser_output["models"])
    print("Hyperparameters:", parser_output["hyperparameters"])

print("Done!")
