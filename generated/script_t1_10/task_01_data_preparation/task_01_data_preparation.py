from dependency import *  # noqa: F401,F403


def data_preparation_1():
    # --------------------------------
    # Load Dataset
    # --------------------------------

    df = pd.read_csv("data.csv")
    print("\nFirst 5 Rows\n")
    print(df.head())

    print("\nDataset Shape:", df.shape)

    # --------------------------------
    # Drop ID Column
    # --------------------------------

    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)
        # Drop empty column if it exists
    if 'Unnamed: 32' in df.columns:
        df.drop('Unnamed: 32', axis=1, inplace=True)

    # --------------------------------
    # Convert Target Column
    # M = 1
    # B = 0
    # --------------------------------

    df['diagnosis'] = df['diagnosis'].map({
        'M': 1,
        'B': 0
    })
    # Remove rows with missing values (if any)
    df.dropna(inplace=True)

    return df
