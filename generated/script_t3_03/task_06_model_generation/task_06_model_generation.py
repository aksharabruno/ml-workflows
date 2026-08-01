from dependency import *  # noqa: F401,F403


def model_generation_6(optimizer):
    step_size = 4
    gamma = 0.2
    scheduler = StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
        )
    return scheduler
