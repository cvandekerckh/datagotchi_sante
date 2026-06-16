"""Lightweight prediction helper for the web application.

The full app.ml.deploy module pulls in the whole training stack (xgboost,
feature engineering, tracking, ...) at import time. The web app only ever needs
to run the already-trained pipeline on a single example, so it imports this
module instead to keep the runtime dependencies minimal (sklearn + pandas).
"""

import pandas as pd

# Prediction column name, kept consistent with app.ml.deploy and the templates.
PREDICTION_COLUMN = "score_tot_prediction"


def predict_for_example(df_example, model_info, is_df_features=True):
    """Predict the well-being score for a single example.

    :param df_example: DataFrame of features (one row).
    :param model_info: tuple (best_model, selected_features).
    :param is_df_features: kept for signature compatibility; must be True here.
    """
    if not is_df_features:
        raise ValueError(
            "webapp_predict.predict_for_example only supports feature DataFrames"
        )
    best_model, selected_features = model_info
    X = df_example[selected_features].values
    y = best_model.predict(X)
    return pd.DataFrame(
        y, index=df_example.index, columns=[PREDICTION_COLUMN]
    )
