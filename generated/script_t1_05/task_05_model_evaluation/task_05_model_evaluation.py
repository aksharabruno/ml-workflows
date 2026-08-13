from dependency import *  # noqa: F401,F403


def model_evaluation_5(rf_model, val_X, val_y):
    rf_val_predictions = rf_model.predict(val_X)
    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

    print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
