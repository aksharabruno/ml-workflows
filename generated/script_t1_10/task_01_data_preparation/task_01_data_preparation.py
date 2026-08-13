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

    # --------------------------------
    # Features and Target
    # --------------------------------

    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    # --------------------------------
    # Train Test Split
    # --------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # --------------------------------
    # Feature Scaling
    # --------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_test, X_train, y_test, y_train
