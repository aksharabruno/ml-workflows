from dependency import *  # noqa: F401,F403


def model_generation_2(lgb_eval, lgb_train):
    # specify your configurations as a dict
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": {"l2", "l1"},
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": 0,
    }

    print("Starting training...")
    # train
    gbm = lgb.train(
        params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, callbacks=[lgb.early_stopping(stopping_rounds=5)]
    )

    print("Saving model...")
    # save model to file
    gbm.save_model("model.txt")

    return gbm
