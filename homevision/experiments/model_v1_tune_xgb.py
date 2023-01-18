import os
from datetime import datetime

import mlflow
import optuna
import xgboost as xgb
from joblib import dump
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from homevision.features.build_features_v1 import (
    apply_features,
    build_features,
    read_data,
)


def __objective(trial: optuna.trial.Trial, *args, **kwargs):

    X_train, y_train = kwargs.get("data", [])[0]
    X_test, y_test = kwargs.get("data", [])[1]

    seed = kwargs.get("seed", 42)
    experiment_id = kwargs.get("experiment_id", None)

    param = {
        "max_depth": trial.suggest_int("max_depth", 1, 64),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 5000),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-3, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.01, 1.0, step=0.01),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.1, 1.0, step=0.01
        ),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        "seed": seed,
    }

    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train)

    metrics = dict(
        train_rmse=mse(y_train, model.predict(X_train), squared=False),
        test_rmse=mse(y_test, model.predict(X_test), squared=False),
        train_mape=mape(y_train, model.predict(X_train)),
        test_mape=mape(y_test, model.predict(X_test)),
    )
    with mlflow.start_run(
        run_name=f"trial-{trial.number}",
        experiment_id=experiment_id,
        tags={"version": "v1"},
        nested=True,
    ):
        for metric in metrics.items():
            mlflow.log_metric(*metric)

    return metrics["test_rmse"]


# TODO: add a trainer for randomForest
# def __objective_rf(trial: optuna.trial.Trial, *args, **kwargs):

#     X_train, y_train = kwargs.get("data", [])[0]
#     X_test, y_test = kwargs.get("data", [])[1]

#     param = {
#         "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
#         "max_depth": trial.suggest_int("max_depth", 1, 32),
#         "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
#         "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 5),
#         # "min_weight_fraction_leaf": float,
#     }

#     model = RandomForestRegressor(**param)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     return mse(y_test, y_pred, squared=False)


def run_experiment():
    """This experiment will try to find the best algorithm among a couple of regressors."""
    identifier = int(datetime.now().timestamp())
    experiment_id = mlflow.create_experiment(
        name=f"XGBRegressor-v1-tuner-{identifier}",
        tags={"version": "v1"},
    )

    seed = 42  # just for reproducibility
    kbins__n_bins = 10
    data = read_data()

    data_train = data[~data.ClosePrice.isna()]

    # to stratify a continouos variable, one option is to discretize it (split it up in bins)
    disc = KBinsDiscretizer(n_bins=kbins__n_bins, encode="ordinal", strategy="quantile")

    # TODO: add cross-validation with stratified k-folds
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

    with mlflow.start_run(
        run_name="optuna-trials",
        experiment_id=experiment_id,
        tags={"version": "v1"},
    ):

        def objective(trial):
            """Current function is a wrapper to pass arguments to __objective.
            (python doesn't allow create inline lambdas, just for that case this function is a common one)
            Args:
                trial (optuna.trial.Trial): an experiment

            Returns:
                float: metric to evaluate
            """

            return __objective(
                trial,
                data=((X_train, y_train), (X_val, y_val)),
                seed=seed,
                experiment_id=experiment_id,
            )

        study = optuna.create_study(
            direction="minimize", study_name="XGBRegressor tuning"
        )
        study.optimize(objective, n_trials=100)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_metric", study.best_value)
        mlflow.log_param("best_trial", str(study.best_trial.number))

    with mlflow.start_run(
        run_name="final-model",
        experiment_id=experiment_id,
        tags={"version": "v1"},
    ):
        data_train, processors = build_features(data_train)  # type: ignore

        model = xgb.XGBRegressor(**study.best_params)
        X, y = (
            data_train.loc[:, ~data_train.columns.isin(["ClosePrice"])],  # type: ignore
            data_train.loc[:, data_train.columns.isin(["ClosePrice"])],  # type: ignore
        )
        model.fit(X, y)

        mlflow.log_metric("rmse", mse(y, model.predict(X), squared=True))
        mlflow.log_metric("mape", mape(y, model.predict(X)))
        mlflow.xgboost.log_model(model, "final-model")

        # save preprocessor as artifacts
        PROCESSORS_PATH = mlflow.get_artifact_uri("processors").replace("file://", "")
        if not os.path.exists(PROCESSORS_PATH):
            os.makedirs(PROCESSORS_PATH)

        for key in ["CDOM", "ElementarySchoolName", "SqFtTotal"]:
            dump(processors.get(key), f"{PROCESSORS_PATH}/{key}.joblib")


if __name__ == "__main__":
    run_experiment()
