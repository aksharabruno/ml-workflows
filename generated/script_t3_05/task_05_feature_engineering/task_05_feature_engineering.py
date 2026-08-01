from dependency import *  # noqa: F401,F403


def feature_engineering_5(ds, label_col, text_col, train_columns):
    def preprocess_fn(examples):
        return tokenizer(
            examples[text_col],
            truncation=True,
            padding=False,
            max_length=256,
        )

    tokenized = ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=[c for c in train_columns if c not in (text_col, label_col)],
    )

