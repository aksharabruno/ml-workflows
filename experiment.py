"""N=5 benchmark runner: run the chunk pipeline over the frozen corpus N times and
report per-stage precision/recall/F1 as mean +/- sd (the citable protocol for the
Evaluation chapter), plus an aggregate confusion matrix for error analysis.

Each run writes to its own experiment_results/run<i>/ directory (so runs don't
overwrite each other); every run is scored against ground_truth.json line by line
(glue-masked, negatives excluded from stage scoring), and the per-stage P/R/F1,
micro F1, ml_problem accuracy and workflow-detection accuracy are aggregated across
runs. A confusion matrix (GT stage x predicted stage, summed over runs) shows where
mislabels go. Results are also written to experiment_results/summary.json.

Usage:
  python experiment.py               # N=5 (default): run the LLM + score
  python experiment.py 10            # N=10 (extend if run-to-run variance is large)
  python experiment.py --score-only  # re-score existing run1..N/ dirs, NO API calls
  python experiment.py 5 --score-only

Running the LLM costs N x (#corpus files) API calls; --score-only costs nothing.
Run in your venv.
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
ABBR = {"environment_configuration": "env", "data_preparation": "dp",
        "feature_engineering": "fe", "model_generation": "mg",
        "model_evaluation": "me", "program_structure": "ps"}
NONE = "(none)"                                    # predicted-nothing bucket for a GT line


def _primary(labels):
    """A scored line carries one stage under the string schema; take a single label
    deterministically (sorted) so the confusion matrix is well-defined even if a
    set ever contains more than one."""
    return sorted(labels)[0] if labels else NONE


def score_run(results_dir: Path, entries, sources):
    """Score one run's results against ground truth. Returns per-stage tp/fp/fn, a
    confusion tally (gt_stage -> pred_stage -> count over scored lines), and the
    ml_problem (exact + coarse) and workflow-detection tallies."""
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))
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
            # single-label confusion: where did this GT line's stage get predicted?
            confusion[_primary(true_labels)][_primary(pred_labels)] += 1

        if positive:                              # ml_problem: positives only
            prob_total += 1
            gt_p, pred_p = entry.get("ml_problem"), result.get("ml_problem")
            prob_exact += (gt_p == pred_p)
            prob_coarse += (coarse_problem(gt_p) == coarse_problem(pred_p))

        wf_total += 1
        wf_correct += (entry.get("is_ml_training_workflow") == result.get("is_ml_training_workflow"))

    return tp, fp, fn, confusion, prob_exact, prob_coarse, prob_total, wf_correct, wf_total


def pr_f1_scores(tp, fp, fn):
    """Per-stage (precision, recall, F1) dict + micro-averaged (P, R, F1)."""
    per_stage = {}
    for s in set(tp) | set(fp) | set(fn):
        p = tp[s] / (tp[s] + fp[s]) if (tp[s] + fp[s]) else 0.0
        r = tp[s] / (tp[s] + fn[s]) if (tp[s] + fn[s]) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        per_stage[s] = (p, r, f)
    T, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    mp = T / (T + FP) if (T + FP) else 0.0
    mr = T / (T + FN) if (T + FN) else 0.0
    mf = 2 * mp * mr / (mp + mr) if (mp + mr) else 0.0
    return per_stage, (mp, mr, mf)


def mean_sd(values):
    m = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0   # sample sd (n-1)
    return m, sd


def print_confusion(confusion_total, n_runs):
    """Print GT-stage x predicted-stage counts, summed over all runs (glue-masked)."""
    cols = [ABBR[s] for s in STAGES] + [NONE]
    keyed = {ABBR[s]: confusion_total.get(s, {}) for s in STAGES}
    print(f"\nConfusion matrix (rows = ground truth, cols = predicted; "
          f"counts summed over {n_runs} runs, glue-masked)")
    header = " " * 6 + "".join(f"{c:>8}" for c in cols)
    print(header)
    print("-" * len(header))
    for s in STAGES:
        row = confusion_total.get(s, {})
        cells = "".join(f"{row.get(cs if cs != NONE else NONE, 0):>8}"
                        for cs in ([k for k in STAGES] + [NONE]))
        print(f"{ABBR[s]:<6}{cells}")


def main(n_runs: int, score_only: bool = False):
    gt = json.loads((REPO / "ground_truth.json").read_text())
    entries = [e for e in gt if e.get("file_type") == "script"
               and (e.get("stage_labels") or not e.get("is_ml_training_workflow", True))]
    sources = {e["file_name"]: (REPO / "test_data" / e["file_name"]).read_text(encoding="utf-8").splitlines()
               for e in entries if (REPO / "test_data" / e["file_name"]).exists()}
    files = list(corpus_files())

    mode = "score-only (no API calls)" if score_only else \
        f"{n_runs} runs x {len(files)} files = {n_runs * len(files)} LLM calls"
    print(f"Benchmark: {mode}.")
    if not score_only:
        print("Ctrl-C to abort.")
    print()

    per_p_runs, per_r_runs, per_f_runs = [], [], []
    micro_p_runs, micro_r_runs, micro_f_runs = [], [], []
    exact_runs, coarse_runs, wf_runs = [], [], []
    confusion_total = defaultdict(lambda: defaultdict(int))

    for i in range(1, n_runs + 1):
        out = REPO / f"experiment_results/run{i}"
        if score_only:
            if not out.exists():
                print(f"=== RUN {i}/{n_runs}  ->  {out.name}/ MISSING — skipped ===")
                continue
            print(f"=== RUN {i}/{n_runs}  ->  {out.name}/ (scoring existing) ===")
        else:
            out.mkdir(parents=True, exist_ok=True)
            pipeline.RESULTS_DIR = out            # redirect this run's output
            print(f"=== RUN {i}/{n_runs}  ->  {out.name}/ ===")
            ok = failed = 0
            for path in files:
                try:
                    run(path, quiet=True)         # extraction only (no --decompose)
                    ok += 1
                except Exception as exc:
                    print(f"  {path.name}: FAILED — {exc}")
                    failed += 1

        tp, fp, fn, conf, exact, coarse, ptot, wc, wt = score_run(out, entries, sources)
        per, (mp, mr, mf) = pr_f1_scores(tp, fp, fn)
        per_p_runs.append({s: v[0] for s, v in per.items()})
        per_r_runs.append({s: v[1] for s, v in per.items()})
        per_f_runs.append({s: v[2] for s, v in per.items()})
        micro_p_runs.append(mp); micro_r_runs.append(mr); micro_f_runs.append(mf)
        exact_runs.append(exact / ptot if ptot else 0.0)
        coarse_runs.append(coarse / ptot if ptot else 0.0)
        wf_runs.append(wc / wt if wt else 0.0)
        for g, row in conf.items():
            for pdt, c in row.items():
                confusion_total[g][pdt] += c
        tag = "scored" if score_only else f"{ok} ok / {failed} failed"
        print(f"  run {i}: {tag} | micro F1 {mf:.3f} | "
              f"ml_problem {exact}/{ptot} | workflow {wc}/{wt}\n")

    # ---- aggregate across runs ----
    print("=" * 72)
    print(f"N={n_runs} AGGREGATE  (mean +/- sample sd)")
    print("=" * 72)
    print(f"{'Stage':<28}{'P':>8}{'sd':>7}{'R':>8}{'sd':>7}{'F1':>8}{'sd':>7}")
    print("-" * 72)
    summary_stages = {}
    for s in STAGES:
        pm, psd = mean_sd([r.get(s, 0.0) for r in per_p_runs])
        rm, rsd = mean_sd([r.get(s, 0.0) for r in per_r_runs])
        fm, fsd = mean_sd([r.get(s, 0.0) for r in per_f_runs])
        print(f"{s:<28}{pm:>8.3f}{psd:>7.3f}{rm:>8.3f}{rsd:>7.3f}{fm:>8.3f}{fsd:>7.3f}")
        summary_stages[s] = {"precision": [pm, psd], "recall": [rm, rsd], "f1": [fm, fsd]}
    print("-" * 72)
    mpm, mpsd = mean_sd(micro_p_runs)
    mrm, mrsd = mean_sd(micro_r_runs)
    mfm, mfsd = mean_sd(micro_f_runs)
    print(f"{'MICRO OVERALL':<28}{mpm:>8.3f}{mpsd:>7.3f}{mrm:>8.3f}{mrsd:>7.3f}{mfm:>8.3f}{mfsd:>7.3f}")

    em, esd = mean_sd(exact_runs); cm, csd = mean_sd(coarse_runs); wm, wsd = mean_sd(wf_runs)
    print(f"\nml_problem (subtype exact): {em:.3f} +/- {esd:.3f}")
    print(f"ml_problem (coarse type):   {cm:.3f} +/- {csd:.3f}")
    print(f"workflow detection:         {wm:.3f} +/- {wsd:.3f}")

    print_confusion(confusion_total, n_runs)

    # ---- machine-readable summary for the thesis / reproducibility ----
    summary = {
        "n_runs": n_runs,
        "model": getattr(pipeline, "MODEL", None) or "see llm_labeller.MODEL",
        "per_stage": summary_stages,
        "micro": {"precision": [mpm, mpsd], "recall": [mrm, mrsd], "f1": [mfm, mfsd]},
        "ml_problem_exact": [em, esd],
        "ml_problem_coarse": [cm, csd],
        "workflow_detection": [wm, wsd],
        "confusion": {g: dict(row) for g, row in confusion_total.items()},
    }
    summary_path = REPO / "experiment_results" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written -> {summary_path.relative_to(REPO)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    score_only = "--score-only" in args
    nums = [a for a in args if a.isdigit()]
    N = int(nums[0]) if nums else 5
    main(N, score_only=score_only)
