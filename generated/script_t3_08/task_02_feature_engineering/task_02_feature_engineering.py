from dependency import *  # noqa: F401,F403


def feature_engineering_2():
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


