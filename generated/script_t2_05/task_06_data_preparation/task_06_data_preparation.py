from dependency import *  # noqa: F401,F403


def data_preparation_6(test_tf):
    img = test_tf(image).unsqueeze(0).to(DEVICE)
    return img
