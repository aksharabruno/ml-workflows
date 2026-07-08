import ast
from pathlib import Path
from stage_map import STAGE_MAP

# Build suffix lookup for resolving library calls
_SUFFIX_MAP: dict[str, str] = {}
for _key, _stage in STAGE_MAP.items():
    parts = _key.split(".")
    for i in range(len(parts)):
        suffix = ".".join(parts[i:])
        if suffix not in _SUFFIX_MAP:
            _SUFFIX_MAP[suffix] = _stage

# Fine-grained stages that map to model_building_and_selection
_MODEL_STAGES = {"model_building", "model_training", "model_parameter_tuning", "model_validation", "train_test_splitting"}
# Fine-grained stages that map to data loading
_DATA_STAGES = {"data_loading"}


def _build_alias_map(tree: ast.AST) -> dict[str, str]:
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
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        parent = _resolve_call(node.value, aliases)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None


def _lookup_stage(resolved: str) -> str | None:
    parts = resolved.split(".")
    for i in range(len(parts)):
        suffix = ".".join(parts[i:])
        if suffix in _SUFFIX_MAP:
            return _SUFFIX_MAP[suffix]
    return None


def _nodes_in_range(tree: ast.AST, start: int, end: int) -> list[ast.AST]:
    return [
        node for node in ast.walk(tree)
        if hasattr(node, "lineno") and start <= node.lineno <= end
    ]


def _extract_hyperparameters(tree: ast.AST, aliases: dict, start: int, end: int) -> list[dict]:
    hyperparameters = []
    for node in _nodes_in_range(tree, start, end):
        if not isinstance(node, ast.Call):
            continue
        resolved = _resolve_call(node.func, aliases)
        if not resolved:
            continue
        stage = _lookup_stage(resolved)
        if stage not in _MODEL_STAGES:
            continue
        kwargs = {}
        for kw in node.keywords:
            if kw.arg and isinstance(kw.value, ast.Constant):
                kwargs[kw.arg] = kw.value.value
        if kwargs:
            hyperparameters.append({
                "name": resolved.split(".")[-1],
                "call": resolved,
                "params": kwargs,
            })
    return hyperparameters


def _extract_dataset(tree: ast.AST, aliases: dict, start: int, end: int) -> dict | None:
    for node in _nodes_in_range(tree, start, end):
        if not isinstance(node, ast.Call):
            continue
        resolved = _resolve_call(node.func, aliases)
        if not resolved:
            continue
        stage = _lookup_stage(resolved)
        if stage not in _DATA_STAGES:
            continue
        # Try to get the first string argument (file path / dataset name)
        source = None
        if node.args and isinstance(node.args[0], ast.Constant):
            source = node.args[0].value
        return {"source": source, "load_call": resolved.split(".")[-1]}
    return None


def _extract_model(tree: ast.AST, aliases: dict, start: int, end: int) -> dict | None:
    for node in _nodes_in_range(tree, start, end):
        if not isinstance(node, ast.Call):
            continue
        resolved = _resolve_call(node.func, aliases)
        if not resolved:
            continue
        stage = _lookup_stage(resolved)
        if stage != "model_building":
            continue
        parts = resolved.split(".")
        return {
            "name": parts[-1],
            "library": parts[0] if len(parts) > 1 else "unknown",
            "full_call": resolved,
        }
    return None


def resolve_calls_by_line(source: str) -> dict[int, list[str]]:
    """Map line number -> fully-qualified resolved call names (for LLM prompt hints)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    aliases = _build_alias_map(tree)
    calls: dict[int, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            resolved = _resolve_call(node.func, aliases)
            if resolved:
                calls.setdefault(node.lineno, []).append(resolved)
    return calls


def extract_from_stages(source: str, stages: list[dict]) -> dict:
    """
    Given source code and LLM-detected stage blocks, extract
    hyperparameters, dataset, and model using AST targeted at the right regions.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"AST parse error: {e}")
        return _empty_extraction()

    aliases = _build_alias_map(tree)

    dataset = None
    model = None
    hyperparameters = []

    for block in stages:
        stage = block.get("stage")
        start = block.get("start", 1)
        end = block.get("end", 1)

        if stage == "data_preparation" and dataset is None:
            dataset = _extract_dataset(tree, aliases, start, end)

        elif stage == "model_generation":
            if model is None:
                model = _extract_model(tree, aliases, start, end)
            hyperparameters.extend(_extract_hyperparameters(tree, aliases, start, end))

    return {
        "dataset": dataset,
        "model": model,
        "hyperparameters": hyperparameters,
    }


def _empty_extraction() -> dict:
    return {
        "dataset": None,
        "model": None,
        "hyperparameters": [],
    }
