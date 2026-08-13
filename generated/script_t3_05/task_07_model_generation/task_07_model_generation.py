from dependency import *  # noqa: F401,F403


def model_generation_7(data_collator, eval_dataset, model, optimizer, train_dataset, training_args):
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

    return trainer
