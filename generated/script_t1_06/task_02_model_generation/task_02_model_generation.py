from dependency import *  # noqa: F401,F403


def model_generation_2(X_train, y_train):
    # 2. تطبيق Linear Regression
    model = LinearRegression()

    # 3. تدريب الموديل
    model.fit(X_train, y_train)

    return model
