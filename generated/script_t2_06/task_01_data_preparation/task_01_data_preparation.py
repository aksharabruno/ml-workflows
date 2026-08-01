from dependency import *  # noqa: F401,F403


def data_preparation_1():
    # Create transform pipeline: Resize images to 224x224 and convert to Tensor
    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToTensor()
    ])

    return transform
