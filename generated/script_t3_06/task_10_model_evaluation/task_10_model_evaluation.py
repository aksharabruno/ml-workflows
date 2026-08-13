from dependency import *  # noqa: F401,F403


def model_evaluation_10(best_model_name, best_score, results):
    comparison_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "accuracy": result["test"]["accuracy"],
                "precision": result["test"]["precision"],
                "recall": result["test"]["recall"],
                "f1": result["test"]["f1"],
                "roc_auc": result["test"]["roc_auc"],
            }
            for model_name, result in results.items()
        ]
    )
    comparison_path = MODELS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    mlflow.set_tag("best_model", best_model_name)
    mlflow.log_metric("best_test_f1", best_score)
    mlflow.log_artifact(str(comparison_path))

