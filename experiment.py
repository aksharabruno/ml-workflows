"""N=5 benchmark runner: run the chunk pipeline over the frozen corpus N times and
report per-stage F1 as mean +/- sd (the citable protocol for the Evaluation chapter).

Each run writes to its own results_run<i>/ directory (so runs don't overwrite each
other); every run is scored against ground_truth.json, and the per-stage F1, micro
F1, ml_problem accuracy and workflow-detection accuracy are aggregated across runs.

Usage:
  python experiment.py         # N=5 (default)
  python experiment.py 10      # N=10 (extend if run-to-run variance is large)

Runs the LLM once per file per run — N x (#corpus files) API calls. Run in your venv.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import pipeline                                   # reused: run(), corpus_files(), RESULTS_DIR
from pipeline import corpus_files, run
from evaluate import is_glue, expand_gt, expand_pred, coarse_problem

REPO = Path(__file__).resolve().parent
STAGES = ["environment_configuration", "data_preparation", "feature_engineering",
          "model_generation", "model_evaluation", "program_structure"]


def score_run(results_dir: Path, entries, sources):
    """Score one run's results against ground truth. Returns per-stage tp/fp/fn plus
    ml_problem (exact + coarse) and workflow-detection tallies."""
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    prob_exact = prob_coarse = prob_total = 0
    wf_correct = wf_total = 0

    for entry in entries:
        stem = Path(entry["file_name"]).stem
        result_path = results_dir / f"{stem}_result.json"
        if not result_path.exists() or entry["file_name"] not in sources:
            continue
        result = json.loads(result_path.read_text())
        source_lines = sources[entry["file_name"]]
        n_lines = len(source_lines)

        positive = entry.get("is_ml_training_workflow", True)
        gt_by_line = expand_gt(entry["stage_labels"]) if entry.get("stage_labels") else {}
        if not positive:                          # negatives: no line scoring
            gt_by_line = {}
        pred_by_line = expand_pred(result.get("stages", []))

        for lineno, true_labels in gt_by_line.items():
            if lineno > n_lines or is_glue(source_lines[lineno - 1]):
                continue
            pred_labels = pred_by_line.get(lineno, set())
            for stage in true_labels:
                (tp if stage in pred_labels else fn)[stage] += 1
            for stage in pred_labels - true_labels:
                fp[stage] += 1

        if positive:                              # ml_problem: positives only
            prob_total += 1
            gt_p, pred_p = entry.get("ml_problem"), result.get("ml_problem")
            prob_exact += (gt_p == pred_p)
            prob_coarse += (coarse_problem(gt_p) == coarse_problem(pred_p))

        wf_total += 1
        wf_correct += (entry.get("is_ml_training_workflow") == result.get("is_ml_training_workflow"))

    return tp, fp, fn, prob_exact, prob_coarse, prob_total, wf_correct, wf_total


def f1_scores(tp, fp, fn):
    """Per-stage F1 dict + micro-averaged F1 from tp/fp/fn tallies."""
    per_stage = {}
    for s in set(tp) | set(fp) | set(fn):
        p = tp[s] / (tp[s] + fp[s]) if (tp[s] + fp[s]) else 0.0
        r = tp[s] / (tp[s] + fn[s]) if (tp[s] + fn[s]) else 0.0
        per_stage[s] = 2 * p * r / (p + r) if (p + r) else 0.0
    T, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    mp = T / (T + FP) if (T + FP) else 0.0
    mr = T / (T + FN) if (T + FN) else 0.0
    micro = 2 * mp * mr / (mp + mr) if (mp + mr) else 0.0
    return per_stage, micro


def mean_sd(values):
    m = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0   # sample sd (n-1)
    return m, sd


def main(n_runs: int):
    gt = json.loads((REPO / "ground_truth.json").read_text())
    entries = [e for e in gt if e.get("file_type") == "script"
               and (e.get("stage_labels") or not e.get("is_ml_training_workflow", True))]
    sources = {e["file_name"]: (REPO / "test_data" / e["file_name"]).read_text(encoding="utf-8").splitlines()
               for e in entries if (REPO / "test_data" / e["file_name"]).exists()}
    files = list(corpus_files())

    print(f"Benchmark: {n_runs} runs x {len(files)} files = {n_runs * len(files)} LLM calls.")
    print("Ctrl-C to abort.\n")

    per_stage_runs, micro_runs = [], []
    exact_runs, coarse_runs, wf_runs = [], [], []

    for i in range(1, n_runs + 1):
        out = REPO / f"experiment_results/run{i}"
        out.mkdir(parents=True, exist_ok=True)
        pipeline.RESULTS_DIR = out                # redirect this run's output
        print(f"=== RUN {i}/{n_runs}  ->  {out.name}/ ===")
        ok = failed = 0
        for path in files:
            try:
                run(path, quiet=True)             # extraction only (no --decompose)
                ok += 1
            except Exception as exc:
                print(f"  {path.name}: FAILED — {exc}")
                failed += 1

        tp, fp, fn, exact, coarse, ptot, wc, wt = score_run(out, entries, sources)
        per, micro = f1_scores(tp, fp, fn)
        per_stage_runs.append(per)
        micro_runs.append(micro)
        exact_runs.append(exact / ptot if ptot else 0.0)
        coarse_runs.append(coarse / ptot if ptot else 0.0)
        wf_runs.append(wc / wt if wt else 0.0)
        print(f"  run {i}: {ok} ok / {failed} failed | micro F1 {micro:.3f} | "
              f"ml_problem {exact}/{ptot} | workflow {wc}/{wt}\n")

    # ---- aggregate across runs ----
    print("=" * 56)
    print(f"N={n_runs} AGGREGATE  (mean +/- sample sd)")
    print("=" * 56)
    print(f"{'Stage':<28} {'F1 mean':>9} {'sd':>7}")
    print("-" * 56)
    for s in STAGES:
        vals = [r.get(s, 0.0) for r in per_stage_runs]
        m, sd = mean_sd(vals)
        print(f"{s:<28} {m:>9.3f} {sd:>7.3f}")
    print("-" * 56)
    m, sd = mean_sd(micro_runs)
    print(f"{'MICRO OVERALL':<28} {m:>9.3f} {sd:>7.3f}")

    em, esd = mean_sd(exact_runs); cm, csd = mean_sd(coarse_runs); wm, wsd = mean_sd(wf_runs)
    print(f"\nml_problem (subtype exact): {em:.3f} +/- {esd:.3f}")
    print(f"ml_problem (coarse type):   {cm:.3f} +/- {csd:.3f}")
    print(f"workflow detection:         {wm:.3f} +/- {wsd:.3f}")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(N)
