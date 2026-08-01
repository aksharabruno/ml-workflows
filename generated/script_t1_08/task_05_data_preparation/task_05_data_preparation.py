from dependency import *  # noqa: F401,F403


def data_preparation_5(scaler):
    # New data prediction
    new_data = np.array([
        [45000, 450, 12000, 1]  # Employment status encoded value
    ])

    # Scale input
    new_scaled = scaler.transform(new_data)

    return new_scaled
