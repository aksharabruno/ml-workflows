from dependency import *  # noqa: F401,F403


def data_preparation_1():
    # Load dataset
    df = pd.read_csv("loan_data.csv")

    # View first 5 rows
    print(df.head())

    # Dataset information
    print(df.info())

    # Encode categorical columns
    le = LabelEncoder()

    df["employment_status"] = le.fit_transform(df["employment_status"])
    df["loan_sanctioned"] = le.fit_transform(df["loan_sanctioned"])

    # Check updated data types
    print(df.info())

    return df
