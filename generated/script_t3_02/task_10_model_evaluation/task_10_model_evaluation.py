from dependency import *  # noqa: F401,F403


def model_evaluation_10(metrics):
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

