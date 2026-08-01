from dependency import *  # noqa: F401,F403


def model_generation_2(X_train, y_train):
    # create model instance
    bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
    # fit model
    bst.fit(X_train, y_train)
    return bst
