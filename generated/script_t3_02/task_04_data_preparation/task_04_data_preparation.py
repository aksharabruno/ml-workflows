from dependency import *  # noqa: F401,F403


def data_preparation_4():
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    required = {
        "first_air_date",
        "name",
        "overview",
        "popularity",
        "vote_average",
        "vote_count",
        "adult",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    y = pd.to_numeric(df["vote_average"], errors="coerce")
    mask = y.notna() & (y >= 0) & (y <= 10)
    df, y = df.loc[mask].reset_index(drop=True), y.loc[mask]

    return df, y
