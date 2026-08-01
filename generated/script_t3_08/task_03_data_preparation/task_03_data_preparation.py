from dependency import *  # noqa: F401,F403


def data_preparation_3(X, df):
    # Fill missing text/categorical values
    text_columns = [
        "title",
        "description",
        "director",
        "country"
    ]

    for col in text_columns:
        X[col] = X[col].fillna("Unknown")

    # Handle missing numerical value
    X["release_year"] = X["release_year"].fillna(
        X["release_year"].median()
    )

    # Target
    y = df["type"]

    # Encode Movie / TV Show -> 0 / 1
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split BEFORE fitting TF-IDF / OneHotEncoder / Scaler
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, label_encoder

    def build_preprocessor():

        preprocessor = ColumnTransformer(
            transformers=[

                # Text features
                (
                    "description_tfidf",
                    TfidfVectorizer(
                        stop_words="english",
                        max_features=5000,
                        ngram_range=(1, 2)
                    ),
                    "description"
                ),

                (
                    "title_tfidf",
                    TfidfVectorizer(
                        stop_words="english",
                        max_features=2000
                    ),
                    "title"
                ),

                # Categorical features
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                    [
                        "director",
                        "country"
                    ]
                ),

                # Numerical feature
                (
                    "numerical",
                    StandardScaler(),
                    ["release_year"]
                )
            ]
        )

        return preprocessor


    return X_test, X_train, label_encoder, y_test, y_train
