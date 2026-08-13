from dependency import *  # noqa: F401,F403


def data_preparation_1():
    TARGET = "type"

    TEXT_COLUMNS = [
        "title",
        "director",
        "cast",
        "country",
        "description"
    ]

    DROP_COLUMNS = [
        "id",
        "date_added",
        "duration",
        "listed_in",
        "platform",
        "rating"
    ]

