import os
from datetime import datetime

import mlflow
import pandas as pd
import xgboost as xg
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeRegressor

from homevision.config.environment import Environment as env
from homevision.features.build_features_v1 import (
    apply_features,
    build_features,
    read_data,
)


def run_experiment():
    """This experiment will try to find the best algorithm among a couple of regressors."""
    identifier = int(datetime.now().timestamp())
    experiment_id = mlflow.create_experiment(
        name=f"select-models-v1-{identifier}",
        tags={"version": "v1"},
    )

    seed = 42  # just for reproducibility
    kbins__n_bins = 10
    data = read_data()

    data_train = data[~data.ClosePrice.isna()]

    # to stratify a continouos variable, one option is to discretize it (split it up in bins)
    # TODO: tune n_bins
    disc = KBinsDiscretizer(n_bins=kbins__n_bins, encode="ordinal", strategy="quantile")

    # TODO: add a kolmorov test to check if a distribution comes from another one.
    # TODO: it will be better to to use stratified k-fold and grid-search than this approach
    df_train, df_eval = train_test_split(
        data_train,
        train_size=0.75,
        random_state=seed,
        stratify=disc.fit_transform(data_train[["ClosePrice"]]).reshape(-1),
    )

    # this method build features and returns processors for its use later.
    df_train, processors = build_features(df_train)  # type: ignore

    # apply the processor to validation data, avoiding data leakage.
    df_eval = apply_features(df_eval, processors)  # type: ignore

    X_train = df_train.loc[:, ~df_train.columns.isin(["ClosePrice"])]  # type: ignore
    y_train = df_train.loc[:, df_train.columns.isin(["ClosePrice"])]  # type: ignore

    X_val = df_eval.loc[:, ~df_eval.columns.isin(["ClosePrice"])]  # type: ignore
    y_val = df_eval.loc[:, df_eval.columns.isin(["ClosePrice"])]  # type: ignore

    rf_regr = RandomForestRegressor(random_state=seed)
    xgb_regr = xg.XGBRegressor(objective="reg:squarederror", seed=seed)
    tree_regr = DecisionTreeRegressor(random_state=seed)
    ridge_reg = Ridge(random_state=seed)
    mlp_regr = MLPRegressor(random_state=seed, verbose=False)

    metrics = {}
    with mlflow.start_run(
        run_name="selecting model",
        experiment_id=experiment_id,
        tags={"version": "v1"},
    ):
        # in a real scenario, we must save the pipeline as an airflow or other graph tool
        # it is necessary if we want reproducibility
        # JUST FOR file:// cases
        PROCESSORS_PATH = mlflow.get_artifact_uri("processors").replace("file://", "")
        if not os.path.exists(PROCESSORS_PATH):
            os.makedirs(PROCESSORS_PATH)

        for key in ["CDOM", "ElementarySchoolName", "SqFtTotal"]:
            dump(processors.get(key), f"{PROCESSORS_PATH}/{key}.joblib")
            # mlflow.log_artifacts(PROCESSORS_PATH)

        for model in [rf_regr, tree_regr, ridge_reg, mlp_regr, xgb_regr]:
            model_id = model.__class__.__name__

            with mlflow.start_run(
                run_name=f"{model_id}",
                experiment_id=experiment_id,
                tags={"version": "v1"},
                nested=True,
            ):
                model.fit(X_train, y_train)
                yt_pred = model.predict(X_train)
                yv_pred = model.predict(X_val)

                metrics[model_id] = dict(
                    train_rmse=mse(y_train, yt_pred, squared=False),
                    train_mae=mae(y_train, yt_pred),
                    train_r2=r2_score(y_train, yt_pred),
                    test_rmse=mse(y_val, yv_pred, squared=False),
                    test_mae=mae(y_val, yv_pred),
                    test_r2=r2_score(y_val, yv_pred),
                )

                # add hyperparams and metrics to tracking system
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics[model_id])

                if "XGB" in model_id:
                    mlflow.xgboost.log_model(model, model_id)
                else:
                    mlflow.sklearn.log_model(model, model_id)

    # I selected rmse because it is more accurate for this case
    # I am able to accept small errors than big ones.
    # I save this file to evaluate the models
    pd.DataFrame(metrics).to_parquet(
        f"{env.FILE_PATH}/raw/selection-models-{experiment_id}.parquet"
    )


if __name__ == "__main__":
    run_experiment()
