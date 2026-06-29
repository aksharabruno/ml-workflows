"""
Evaluate static and LLM stage predictions against ground truth.

Reads results/<stem>_result.json for each script entry in ground_truth.json.
Reports precision, recall, F1 per stage label + micro-averaged overall,
separately for static analysis and LLM analysis.

Comparison unit: file × stage (binary — was this stage detected in this file?)
  TP: stage in ground truth AND predicted
  FP: stage predicted but NOT in ground truth
  FN: stage in ground truth but NOT predicted
"""

import json
from collections import defaultdict
from pathlib import Path

GROUND_TRUTH_PATH = Path("ground_truth.json")
OUTPUT_DIR = Path("results")


def gt_stages(entry: dict) -> set[str]:
    stages = set()
    for labels in entry["stage_labels"].values():
        stages.update(labels)
    return stages


def llm_stages(result: dict) -> set[str]:
    stage_labels = result.get("llm_analysis", {}) or {}
    stage_labels = stage_labels.get("stage_labels", {}) or {}
    stages = set()
    for labels in stage_labels.values():
        stages.update(labels)
    return stages


def score(true: set, pred: set, tp, fp, fn):
    for stage in true:
        if stage in pred:
            tp[stage] += 1
        else:
            fn[stage] += 1
    for stage in pred:
        if stage not in true:
            fp[stage] += 1


def print_table(tp, fp, fn, title: str):
    all_labels = sorted(tp.keys() | fp.keys() | fn.keys())
    if not all_labels:
        print(f"\n{title}: no data.\n")
        return

    col = 36
    print(f"\n{'─' * (col + 32)}")
    print(f"  {title}")
    print(f"{'─' * (col + 32)}")
    print(f"{'Stage':<{col}} {'P':>6} {'R':>6} {'F1':>6}   {'TP':>3} {'FP':>3} {'FN':>3}")
    print(f"{'─' * (col + 32)}")

    total_tp = total_fp = total_fn = 0
    for label in all_labels:
        t, f_p, f_n = tp[label], fp[label], fn[label]
        p  = t / (t + f_p) if (t + f_p) > 0 else 0.0
        r  = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"{label:<{col}} {p:>6.2f} {r:>6.2f} {f1:>6.2f}   {t:>3} {f_p:>3} {f_n:>3}")
        total_tp += t
        total_fp += f_p
        total_fn += f_n

    micro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    print(f"{'─' * (col + 32)}")
    print(f"{'MICRO OVERALL':<{col}} {micro_p:>6.2f} {micro_r:>6.2f} {micro_f1:>6.2f}   {total_tp:>3} {total_fp:>3} {total_fn:>3}")


def evaluate():
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)

    entries = [
        e for e in ground_truth
        if e["file_type"] == "script" and e.get("stage_labels")
    ]

    static_tp: dict[str, int] = defaultdict(int)
    static_fp: dict[str, int] = defaultdict(int)
    static_fn: dict[str, int] = defaultdict(int)

    llm_tp: dict[str, int] = defaultdict(int)
    llm_fp: dict[str, int] = defaultdict(int)
    llm_fn: dict[str, int] = defaultdict(int)

    evaluated = 0
    print(f"Evaluating {len(entries)} script entries...\n")

    for entry in entries:
        stem = Path(entry["file_name"]).stem
        result_path = OUTPUT_DIR / f"{stem}_result.json"

        if not result_path.exists():
            print(f"  [skip] {entry['file_name']} — run main.py first.")
            continue

        with open(result_path) as f:
            result = json.load(f)

        true = gt_stages(entry)
        static_pred = set(result["static_analysis"]["stages_detected"].keys())
        llm_pred = llm_stages(result)

        score(true, static_pred, static_tp, static_fp, static_fn)
        score(true, llm_pred, llm_tp, llm_fp, llm_fn)
        evaluated += 1

    print_table(static_tp, static_fp, static_fn, "Static Analysis (STAGE_MAP)")
    print_table(llm_tp, llm_fp, llm_fn, "LLM Analysis")
    print(f"\nFiles evaluated: {evaluated} / {len(entries)}")


if __name__ == "__main__":
    evaluate()
