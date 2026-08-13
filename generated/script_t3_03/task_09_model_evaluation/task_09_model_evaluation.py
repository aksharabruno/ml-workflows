from dependency import *  # noqa: F401,F403


def model_evaluation_9(TestLoader, modelnew):
    modelnew.eval()

    modelnew.eval()
    with torch.no_grad():
        all_probs = torch.tensor([])
        for images, labels in tqdm(TestLoader, desc = "Testing on unseen data", leave = False):
            pred = modelnew(images)
            pred_out1 = torch.nn.functional.softmax(pred, dim = 1)
            all_probs = torch.cat((all_probs, pred_out1),dim = 0)

    predictions = torch.argmax(all_probs, dim = 1)

    return predictions
