from dependency import *  # noqa: F401,F403


def data_preparation_2(NUMERIC_COLS):
    parts: list[tuple] = [
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            NUMERIC_COLS,
        ),
        ("bin", "passthrough", ["adult"]),
    ]
    return parts
