from dependency import *  # noqa: F401,F403


def model_generation_9(data_collator, eval_dataset, model, optimizer, train_dataset, training_args):
    print("Creating Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None) if optimizer is not None else (None, None),
    )

    # Print sample GPU memory (nvidia-smi) and PyTorch summary if GPU available
    if torch.cuda.is_available():
        try:
            print("nvidia-smi output (summary):")
            os.system("nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv -l 1 -n 1")
        except Exception:
            pass

    # Train
    print("Starting training...")
    trainer.train()

    return trainer
