from dependency import *  # noqa: F401,F403


def model_evaluation_7(ls, x_test, y_test):
    test_data_prediction = ls.predict(x_test)

    test_data_prediction= ls.predict(x_test)

    error_score = metrics.r2_score(y_test, test_data_prediction)
    print("R squared Error:", error_score)

    plt.figure()
    plt.scatter(y_test, test_data_prediction, color='crimson', alpha=0.7)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Lasso Regression – Test Data")



    return test_data_prediction
