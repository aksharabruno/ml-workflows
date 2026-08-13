from dependency import *  # noqa: F401,F403


def model_generation_3(X_train, y_train):
    results = {}
    best_score = -1.0
    best_model_name = ""
    best_pipeline = None
    best_metrics = None

    for model_name in ["logistic_regression", "random_forest", "xgboost", "catboost"]:
        with mlflow.start_run(run_name=model_name):
            if model_name == "random_forest":
                param_grid = {"model__n_estimators": [50, 100], "model__max_depth": [5, 10]}
                pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(random_state=42, class_weight="balanced"))])
                search = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="f1", n_jobs=-1)
                search.fit(X_train, y_train)
                pipeline = search.best_estimator_
            else:
                pipeline = _build_pipeline(model_name)
                pipeline.fit(X_train, y_train)

    return best_model_name, best_pipeline, best_score, model_name, pipeline, results
