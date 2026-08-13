from dependency import *  # noqa: F401,F403


def model_generation_10(model):
    # Save PEFT adapter weights only (small)
    print("Saving PEFT adapter to:", args.output_dir)
    model.save_pretrained(args.output_dir)

    print("Done.")

