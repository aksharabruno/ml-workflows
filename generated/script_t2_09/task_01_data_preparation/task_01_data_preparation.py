from dependency import *  # noqa: F401,F403


def data_preparation_1():
    # 2. Generate Dataset
    print("Step 2: Preparing text database...")
    texts, labels = generate_synthetic_dataset()
    print(f"  Total samples generated: {len(texts)}")
    print(f"  Classes: {CLASSES}\n")

    return labels, texts
