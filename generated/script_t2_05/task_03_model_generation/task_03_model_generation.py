from dependency import *  # noqa: F401,F403


def model_generation_3():
    model = FlowerModel().to(DEVICE)
    return model
