from dependency import *  # noqa: F401,F403


def model_evaluation_11(classes, modelnew, predictions, targets_val, transformNorm):

    precision = precision_score(targets_val, predictions, average='macro')
    recall = recall_score(targets_val, predictions, average='macro')

    print("Precision:", precision)
    print("Recall:", recall)

    image.shape

    image_path = r"C:\Users\Jelil\Desktop\New folder (7)\seg_test\seg_test\buildings\20057.jpg"
    image = Image.open(image_path)
    image

    image = transformNorm(image)
    image = image.unsqueeze(0)
    modelnew.eval()
    with torch.no_grad():
        pred = modelnew(image)
        pred_out1 = torch.nn.functional.softmax(pred, dim = 1)
        prediction = torch.argmax(pred_out1, dim = 1)
    print(classes[prediction.item()])

