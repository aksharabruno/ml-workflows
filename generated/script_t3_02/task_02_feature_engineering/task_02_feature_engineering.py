from dependency import *  # noqa: F401,F403


def feature_engineering_2(parts):
    if text_features:
        parts.append(
            (
                "text",
                Pipeline(
                    [
                        (
                            "tfidf",
                            TfidfVectorizer(
                                max_features=500,
                                min_df=1,
                                ngram_range=(1, 2),
                                sublinear_tf=True,
                            ),
                        ),
                        ("svd", TruncatedSVD(n_components=48, random_state=42)),
                    ]
                ),
                "text_blob",
            )
        )
