from dependency import *  # noqa: F401,F403


def feature_engineering_2(red_wine, white_wine):
    # Add wine type feature
    red_wine["wine_type"] = 0   # Red
    white_wine["wine_type"] = 1 # White

