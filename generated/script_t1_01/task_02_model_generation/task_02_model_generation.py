from dependency import *  # noqa: F401,F403


def model_generation_2(x_train_scaled, y_train):
    # model training
    print("Training a random forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train_scaled, y_train)

    return model
