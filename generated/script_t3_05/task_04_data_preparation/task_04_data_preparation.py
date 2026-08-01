from dependency import *  # noqa: F401,F403


def data_preparation_4(label_col, text_col, train_columns):
    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not infer text/label columns. Found columns: {train_columns}"
        )

