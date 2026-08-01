from dependency import *  # noqa: F401,F403


def model_evaluation_5(X, mae, model, r2):
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R²): {r2:.2f}")
    print("\nModel Coefficients (Weights):")
    for col, coef in zip(X.columns, model.coef_):
        print(f"{col}: {coef:.2f}")
