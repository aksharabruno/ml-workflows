from dependency import *  # noqa: F401,F403


def model_evaluation_3(lr, x_test, x_train, y_test, y_train):
    training_data_prediction = lr.predict(x_train)

    eror_score=  metrics.r2_score(y_train, training_data_prediction)
    print("R squared Error : ", eror_score)

    plt.figure()
    plt.scatter(y_train, training_data_prediction, color='steelblue', alpha=0.7)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Linear Regression – Training Data")


    test_data_prediction = lr.predict(x_test)

    error_score = metrics.r2_score(y_test, test_data_prediction)
    print("R squared Error : ", error_score)

    plt.figure()
    plt.scatter(y_test, test_data_prediction, color='darkorange', alpha=0.7)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Linear Regression – Test Data")


    return test_data_prediction
