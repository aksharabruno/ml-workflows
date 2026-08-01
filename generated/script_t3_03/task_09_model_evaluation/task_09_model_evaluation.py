from dependency import *  # noqa: F401,F403


def model_evaluation_9(TestLoader, classes, predictions):
    targets_val = torch.cat([labels for _, labels in tqdm(TestLoader, desc="Get Labels", leave = False)])

    cm = confusion_matrix(targets_val, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes.values())
    # Set figure size

    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")

    return targets_val
