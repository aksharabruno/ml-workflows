from dependency import *  # noqa: F401,F403


def model_evaluation_5(ls, x_train, y_train):
    training_data_prediction = ls.predict(x_train)

    error_score = metrics.r2_score(y_train, training_data_prediction)
    print("R squared Error:", error_score)

    plt.figure()
    plt.scatter(y_train, training_data_prediction, color='seagreen', alpha=0.7)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Lasso Regression – Training Data")


