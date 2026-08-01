from dependency import *  # noqa: F401,F403


def feature_engineering_3(candidate_label_cols, candidate_text_cols, train_columns):
    text_col = next((c for c in candidate_text_cols if c in train_columns), None)
    label_col = next((c for c in candidate_label_cols if c in train_columns), None)

    return label_col, text_col
