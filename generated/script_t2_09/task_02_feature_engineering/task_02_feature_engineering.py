from dependency import *  # noqa: F401,F403


def feature_engineering_2(texts):
    # 3. Tokenize Dataset
    print("Step 3: Tokenizing text inputs for BERT...")
    encoded = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='np'
    )

    X_ids = encoded['input_ids'].astype(np.int32)
    X_mask = encoded['attention_mask'].astype(np.int32)
    X_type = encoded['token_type_ids'].astype(np.int32)

    return X_ids, X_mask, X_type
