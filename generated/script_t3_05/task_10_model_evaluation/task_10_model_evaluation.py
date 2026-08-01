from dependency import *  # noqa: F401,F403


def model_evaluation_10(trainer):
    print("Evaluating...")
    eval_res = trainer.evaluate()
    return eval_res
