from dependency import *  # noqa: F401,F403


def data_preparation_1():
    # Load the data, and separate the target
    iowa_file_path = '../input/train.csv'
    home_data = pd.read_csv(iowa_file_path)
    y = home_data.SalePrice

    return home_data, y
