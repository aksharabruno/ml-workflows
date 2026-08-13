from dependency import *  # noqa: F401,F403


def data_preparation_4(tokenized):
    train_dataset = tokenized["train"]
    eval_dataset = tokenized["test"]

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return data_collator, eval_dataset, train_dataset
