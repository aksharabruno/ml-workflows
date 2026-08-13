from dependency import *  # noqa: F401,F403


def data_preparation_3():
    # Load data


    df = load_data(
        args.data_path
    )



    # EDA


    perform_eda(df)



    return df
