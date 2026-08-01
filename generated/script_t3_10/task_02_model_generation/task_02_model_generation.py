from dependency import *  # noqa: F401,F403


def model_generation_2():
    model = BertModel.from_pretrained('bert-base-uncased')

    return model
