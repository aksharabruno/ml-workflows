from dependency import *  # noqa: F401,F403


def model_generation_9(X_train_res, grid_search, y_train_res):
    # Extract the best parameters without the 'svc__' prefix
    best_params = {k.replace('svc__', ''): v for k, v in grid_search.best_params_.items()}

    print("Best parameters found: ", best_params)

    # Train the final SVM classifier with the best parameters
    print("Training the final SVM classifier...")
    final_svm_model = make_pipeline(StandardScaler(), SVC(decision_function_shape='ovr', probability=True, random_state=42, **best_params))
    final_svm_model.fit(X_train_res, y_train_res)

    return final_svm_model
