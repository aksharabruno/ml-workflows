from dependency import *  # noqa: F401,F403


def data_preparation_2(base_dir, downloadedfile):
    with zipfile.ZipFile(downloadedfile, "r") as f:
        f.extractall(base_dir)

    height = 224
    width = 224

    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()
        ])

    dataset_path = r"C:\Users\Jelil\Desktop\New folder (7)\seg_train\seg_train"
    datasetz = datasets.ImageFolder(root = dataset_path, transform = transform)

    classes = {v:k for k,v in datasetz.class_to_idx.items()}
    return classes, dataset_path, datasetz
