from dependency import *  # noqa: F401,F403


def model_generation_1():
    def _build_model(name: str):
        if name == "logistic_regression":
            return LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")
        if name == "random_forest":
            return RandomForestClassifier(random_state=42, class_weight="balanced")
        if name == "xgboost":
            return XGBClassifier(random_state=42, eval_metric="logloss", n_estimators=100, scale_pos_weight=2.5)
        if name == "catboost":
            return CatBoostClassifier(verbose=False, random_state=42, iterations=200, auto_class_weights="Balanced")
        raise ValueError(f"Unknown model: {name}")
