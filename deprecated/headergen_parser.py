import subprocess
import json
from pathlib import Path
from stage_map import STAGE_MAP

def generate_headergen_annotations(input_file, output_dir):
    try:
        subprocess.run(
            ["headergen", "generate", "-i", str(input_file), "-o", str(output_dir), "-j"],
            check=True,
        )
        print(f"Success: {input_file.name}")
    except subprocess.CalledProcessError:
        print(f"Failed: {input_file.name}")

def parse_headergen_output(json_path):
    with open(json_path) as f:
        data = json.load(f)

    cell_callsites = data.get("cell_callsites", {})
    context_calls = data.get("analysis_info", {}).get("context_library_calls", {})

    detected_stages = {}
    dataset_sources = []
    models = []
    hyperparameters = []

    # Build a lookup: function name → kwargs from analysis_info
    kwargs_lookup = {}
    call_args = data.get("analysis_info", {}).get("call_args", {})
    for key, entry in call_args.items():
        # key looks like "sklearn.ensemble._forest.RandomForestClassifier:1125"
        func_name = key.rsplit(":", 1)[0]
        kwargs = entry.get("kwargs", {})
        if kwargs:
            # Store all kwargs per function, there may be multiple calls
            kwargs_lookup.setdefault(func_name, []).append(kwargs)

    #print("kwargs_lookup keys:", list(kwargs_lookup.keys()))

    # Walk cell callsites and apply STAGE_MAP
    for cell_id, calls in cell_callsites.items():
        for call in calls:
            if call in STAGE_MAP:
                stage = STAGE_MAP[call]
                detected_stages.setdefault(stage, []).append(call)

                if stage == "data_loading":
                    dataset_sources.append(call)

                elif stage == "model_building":
                    if call not in models:
                        models.append(call)
                    for kw in kwargs_lookup.get(call, []):
                        hyperparameters.append({
                            "call": call,
                            "stage": "model_building",
                            "kwargs": kw
                    })

                elif stage == "train_test_splitting":
                    for kw in kwargs_lookup.get(call, []):
                        hyperparameters.append({
                        "call": call,
                        "stage": "train_test_splitting",
                        "kwargs": kw
                })


    is_ml_training_workflow = all(
        stage in detected_stages
        for stage in ["model_building", "model_training"]
    )

    return {
        "is_ml_training_workflow": is_ml_training_workflow,
        "dataset": dataset_sources,
        "models": models,
        "hyperparameters": hyperparameters,
        "stages_detected": detected_stages,
    }