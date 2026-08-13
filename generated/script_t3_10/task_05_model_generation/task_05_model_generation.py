from dependency import *  # noqa: F401,F403


def model_generation_5(X_train_subset_res, y_train_subset_res):
    # Hyperparameter tuning for SVM with RBF and Polynomial kernels
    print("Tuning SVM hyperparameters...")
    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['rbf', 'poly'],
        'svc__gamma': ['scale', 'auto'],
        'svc__degree': [3, 4, 5]  # Only relevant for polynomial kernel
    }
    svm_model = make_pipeline(StandardScaler(), SVC(decision_function_shape='ovr', probability=True, random_state=42))
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_subset_res, y_train_subset_res)

    best_params = grid_search.best_params_
    print("Best parameters found: ", best_params)

    return grid_search
