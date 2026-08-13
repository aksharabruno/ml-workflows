from dependency import *  # noqa: F401,F403


def model_evaluation_8(lr, test_data_prediction, x_test, y_test):
    # Linear Regression test predictions
    lr_test_prediction = lr.predict(x_test)

    # Lasso Regression test predictions
    # In the current notebook state, 'test_data_prediction' holds the Lasso predictions on x_test.
    lasso_test_prediction = test_data_prediction

    # Calculate residuals (errors)
    lr_residuals = y_test - lr_test_prediction
    lasso_residuals = y_test - lasso_test_prediction

    plt.figure(figsize=(14, 6))

    # Plot histogram for Linear Regression residuals
    plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
    sns.histplot(lr_residuals, kde=True, color='blue', bins=30)
    plt.title('Distribution of Linear Regression Errors (Test Data)')
    plt.xlabel('Residuals (Actual Price - Predicted Price)')
    plt.ylabel('Frequency')

    # Plot histogram for Lasso Regression residuals
    plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
    sns.histplot(lasso_residuals, kde=True, color='red', bins=30)
    plt.title('Distribution of Lasso Regression Errors (Test Data)')
    plt.xlabel('Residuals (Actual Price - Predicted Price)')
    plt.ylabel('Frequency')

    plt.tight_layout()

