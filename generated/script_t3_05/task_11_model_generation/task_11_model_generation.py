from dependency import *  # noqa: F401,F403


def model_generation_11(eval_res, model):
    print("Eval results:", eval_res)

    # Save PEFT adapter weights only (small)
    print("Saving PEFT adapter to:", args.output_dir)
    model.save_pretrained(args.output_dir)

