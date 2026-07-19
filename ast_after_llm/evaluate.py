"""
Evaluate pipeline results against ground_truth.json.

Reads results/<stem>_result.json for each script entry in the ground truth.

Stage scoring unit: line × stage.
  - Only lines covered by a ground-truth range are scored (predictions are
    gap-free by design, so unlabeled ground-truth lines carry no signal).
  - Glue lines (blank, comment-only, bare print) are masked out so boundary
    placement inside glue never counts for or against the model.
  - Ranges are clamped to the actual file length.

Also reports file-level accuracy for ml_problem and is_ml_training_workflow.
"""

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_PATH = REPO_ROOT / "ground_truth.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TEST_DATA_DIR = REPO_ROOT / "test_data"


def is_glue(line: str) -> bool:
    stripped = line.strip()
    return (
        not stripped
        or stripped.startswith("#")
        or stripped.startswith("print(")
        or stripped.startswith("print ")
    )


def expand_gt(stage_labels: dict) -> dict[int, set[str]]:
    """Expand {"1-5": [...], "7": [...]} to per-line label sets."""
    by_line: dict[int, set[str]] = defaultdict(set)
    for key, labels in stage_labels.items():
        parts = key.split("-")
        start = int(parts[0])
        end = int(parts[1]) if len(parts) > 1 else start
        for lineno in range(start, end + 1):
            by_line[lineno].update(labels)
    return by_line


def expand_pred(stages: list) -> dict[int, set[str]]:
    """Expand result "stages" blocks to per-line label sets."""
    by_line: dict[int, set[str]] = defaultdict(set)
    for block in stages:
        for lineno in range(block["start"], block["end"] + 1):
            by_line[lineno].add(block["stage"])
    return by_line


def print_table(tp, fp, fn, title: str):
    all_labels = sorted(tp.keys() | fp.keys() | fn.keys())
    if not all_labels:
        print(f"\n{title}: no data.\n")
        return

    col = 36
    print(f"\n{'─' * (col + 32)}")
    print(f"  {title}")
    print(f"{'─' * (col + 32)}")
    print(f"{'Stage':<{col}} {'P':>6} {'R':>6} {'F1':>6}   {'TP':>4} {'FP':>4} {'FN':>4}")
    print(f"{'─' * (col + 32)}")

    total_tp = total_fp = total_fn = 0
    for label in all_labels:
        t, f_p, f_n = tp[label], fp[label], fn[label]
        p = t / (t + f_p) if (t + f_p) > 0 else 0.0
        r = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"{label:<{col}} {p:>6.2f} {r:>6.2f} {f1:>6.2f}   {t:>4} {f_p:>4} {f_n:>4}")
        total_tp += t
        total_fp += f_p
        total_fn += f_n

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    print(f"{'─' * (col + 32)}")
    print(f"{'MICRO OVERALL':<{col}} {micro_p:>6.2f} {micro_r:>6.2f} {micro_f1:>6.2f}   {total_tp:>4} {total_fp:>4} {total_fn:>4}")


def evaluate():
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)

    entries = [
        e for e in ground_truth
        if e["file_type"] == "script" and e.get("stage_labels")
    ]

    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    problem_correct = 0
    workflow_correct = 0
    evaluated = 0

    print(f"Evaluating {len(entries)} script entries...\n")
    print(f"{'File':<22} {'lines':>5} {'match':>5} {'acc':>6}   {'ml_problem':<28} {'wf':>3}")
    print("─" * 78)

    for entry in entries:
        stem = Path(entry["file_name"]).stem
        result_path = RESULTS_DIR / f"{stem}_result.json"
        source_path = TEST_DATA_DIR / entry["file_name"]

        if not result_path.exists():
            print(f"{entry['file_name']:<22} [skip] no result — run main.py first.")
            continue
        if not source_path.exists():
            print(f"{entry['file_name']:<22} [skip] source not found in test_data/.")
            continue

        with open(result_path) as f:
            result = json.load(f)
        source_lines = source_path.read_text(encoding="utf-8").splitlines()
        n_lines = len(source_lines)

        gt_by_line = expand_gt(entry["stage_labels"])
        pred_by_line = expand_pred(result.get("stages", []))

        file_agree = 0
        file_scored = 0
        for lineno, true_labels in sorted(gt_by_line.items()):
            if lineno > n_lines or is_glue(source_lines[lineno - 1]):
                continue
            pred_labels = pred_by_line.get(lineno, set())
            for stage in true_labels:
                if stage in pred_labels:
                    tp[stage] += 1
                else:
                    fn[stage] += 1
            for stage in pred_labels - true_labels:
                fp[stage] += 1
            file_scored += 1
            if pred_labels == true_labels:
                file_agree += 1

        acc = file_agree / file_scored if file_scored else 0.0

        gt_problem = entry.get("ml_problem", "unknown")
        pred_problem = result.get("ml_problem", "unknown")
        problem_ok = gt_problem == pred_problem
        problem_correct += problem_ok

        wf_ok = entry.get("is_ml_training_workflow") == result.get("is_ml_training_workflow")
        workflow_correct += wf_ok

        problem_str = pred_problem + ("" if problem_ok else f" (gt: {gt_problem})")
        print(f"{entry['file_name']:<22} {file_scored:>5} {file_agree:>5} {acc:>6.2f}   {problem_str:<28} {'ok' if wf_ok else 'X':>3}")
        evaluated += 1

    print_table(tp, fp, fn, "Stage labeling (per line, glue masked)")

    if evaluated:
        print(f"\nml_problem accuracy:          {problem_correct}/{evaluated}")
        print(f"is_ml_training_workflow:      {workflow_correct}/{evaluated}")
    print(f"\nFiles evaluated: {evaluated} / {len(entries)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:  # optional: score a different results dir (ablations)
        _p = Path(sys.argv[1])
        RESULTS_DIR = _p if _p.is_absolute() else REPO_ROOT / _p
    evaluate()
