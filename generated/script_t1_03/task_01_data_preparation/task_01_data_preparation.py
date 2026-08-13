from dependency import *  # noqa: F401,F403


def data_preparation_1():
    print("Loading data...")
    # load or create your dataset
    regression_example_dir = Path(__file__).absolute().parents[1] / "regression"
    df_train = pd.read_csv(str(regression_example_dir / "regression.train"), header=None, sep="\t")
    df_test = pd.read_csv(str(regression_example_dir / "regression.test"), header=None, sep="\t")

    y_train = df_train[0]
    y_test = df_test[0]
    X_train = df_train.drop(0, axis=1)
    X_test = df_test.drop(0, axis=1)

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    return X_test, lgb_eval, lgb_train, y_test
