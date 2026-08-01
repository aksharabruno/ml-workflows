from dependency import *  # noqa: F401,F403


def model_generation_6(y_prob):
    # --------------------------------
    # Threshold Example
    # --------------------------------

    threshold = 0.60

    custom_prediction = (y_prob >= threshold).astype(int)

    return custom_prediction, threshold
