from dependency import *  # noqa: F401,F403


def data_preparation_1():
    # --- 3. Chargement des données ---
    texts, labels, label_names = load_bbc_dataset(args.data_path)
    return label_names, labels, texts
