from dependency import *  # noqa: F401,F403


def model_evaluation_7(img, model):
    with torch.no_grad():
        probs = F.softmax(model(img), dim=1)[0]
        top3 = torch.topk(probs, k=3)
    predictions = [(CLASSES[idx], float(prob)) for idx, prob in zip(top3.indices, top3.values)]
    return predictions

