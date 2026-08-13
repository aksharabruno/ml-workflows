from dependency import *  # noqa: F401,F403


def model_evaluation_9(trainer):
    print("Evaluating...")
    eval_res = trainer.evaluate()
    print("Eval results:", eval_res)

