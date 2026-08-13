from dependency import *  # noqa: F401,F403


def data_preparation_1():
        # data generation
    print("Generating sample data...")
    x = np.random.randn(1000, 20)
    y = np.random.randint(0, 2, 1000)

    # data preparation
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_test_scaled, x_train_scaled, y_test, y_train
