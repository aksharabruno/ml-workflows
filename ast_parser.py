import ast
from pathlib import Path
from stage_map import STAGE_MAP

# Build suffix lookup: "read_csv" -> stage, "pandas.read_csv" -> stage, etc.
# Allows matching without full qualification
_SUFFIX_MAP: dict[str, str] = {}
for _key, _stage in STAGE_MAP.items():
    parts = _key.split(".")
    for i in range(len(parts)):
        suffix = ".".join(parts[i:])
        if suffix not in _SUFFIX_MAP:
            _SUFFIX_MAP[suffix] = _stage


def _build_alias_map(tree: ast.AST) -> dict[str, str]:
    """Map local alias -> canonical module prefix.
    e.g. 'import pandas as pd'  ->  {'pd': 'pandas'}
         'from sklearn.preprocessing import StandardScaler'  -> {'StandardScaler': 'sklearn.preprocessing.StandardScaler'}
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                aliases[local] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                local = alias.asname or alias.name
                aliases[local] = f"{node.module}.{alias.name}"
    return aliases


def _resolve_call(node: ast.expr, aliases: dict[str, str]) -> str | None:
    """Turn a Call's func node into a dotted string using import aliases."""
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        parent = _resolve_call(node.value, aliases)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None


def _lookup_stage(resolved: str) -> str | None:
    """Try progressively shorter suffixes against _SUFFIX_MAP."""
    parts = resolved.split(".")
    for i in range(len(parts)):
        suffix = ".".join(parts[i:])
        if suffix in _SUFFIX_MAP:
            return _SUFFIX_MAP[suffix]
    return None


def parse_file(file_path: Path) -> dict:
    source = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"AST parse error: {e}")
        return _empty_result()

    aliases = _build_alias_map(tree)
    lines = source.splitlines()

    stage_labels: dict[str, list[str]] = {}  # "lineno": [stage, ...]
    detected_stages: dict[str, list[str]] = {}  # stage: [call, ...] for summary
    dataset_sources: list[str] = []
    models: list[str] = []
    hyperparameters: list[dict] = []

    # library_loading: any import line
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            key = str(node.lineno)
            stage_labels.setdefault(key, [])
            if "library_loading" not in stage_labels[key]:
                stage_labels[key].append("library_loading")
            detected_stages.setdefault("library_loading", [])

    # Walk all Call nodes
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        resolved = _resolve_call(node.func, aliases)
        if not resolved:
            continue

        stage = _lookup_stage(resolved)
        if not stage:
            continue

        label = resolved.split(".")[-1]
        lineno = str(getattr(node, "lineno", "?"))
        stage_labels.setdefault(lineno, [])
        if stage not in stage_labels[lineno]:
            stage_labels[lineno].append(stage)
        detected_stages.setdefault(stage, []).append(label)

        if stage == "data_loading":
            dataset_sources.append(resolved)

        elif stage == "model_building":
            if resolved not in models:
                models.append(resolved)
            kwargs = {kw.arg: ast.literal_eval(kw.value)
                      for kw in node.keywords
                      if kw.arg and isinstance(kw.value, ast.Constant)}
            if kwargs:
                hyperparameters.append({"call": resolved, "stage": "model_building", "kwargs": kwargs})

        elif stage == "train_test_splitting":
            kwargs = {kw.arg: ast.literal_eval(kw.value)
                      for kw in node.keywords
                      if kw.arg and isinstance(kw.value, ast.Constant)}
            if kwargs:
                hyperparameters.append({"call": resolved, "stage": "train_test_splitting", "kwargs": kwargs})

    is_ml = all(s in detected_stages for s in ["model_building", "model_training"])

    return {
        "is_ml_training_workflow": is_ml,
        "dataset": dataset_sources,
        "models": models,
        "hyperparameters": hyperparameters,
        "stages_detected": detected_stages,
        "stage_labels": dict(sorted(stage_labels.items(), key=lambda x: int(x[0]))),
    }


def _empty_result() -> dict:
    return {
        "is_ml_training_workflow": False,
        "dataset": [],
        "models": [],
        "hyperparameters": [],
        "stages_detected": {},
        "stage_labels": {},
    }
