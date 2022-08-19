import pickle
import os

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

RS = 2022

OPTUNA_EXPERIMENT_NAME = "xgboost-optuna"
EXPERIMENT_NAME = "xgboost-best-models"

mlflow.set_tracking_uri("sqlite:///mlflow-locally.db")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.xgboost.autolog()


@task
def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@task
def train_best_model(train, valid, y_val, dv, params):
    with mlflow.start_run():

        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=10
        )

        y_pred = booster.predict(valid)
        validation_rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("validation_rmse", validation_rmse)

        with open("models/preprocessor_xgb.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor_xgb.b",
                            artifact_path="preprocessor_xgb")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow_xgb")


@flow(task_runner=SequentialTaskRunner())
def train(data_path: str = "./preprocess_data"):

    # Load data
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "test.pkl"))
    dv = load_pickle(os.path.join(data_path, "dv.pkl"))

    # Prepare data for xgboost
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    # Run mlflow client
    client = MlflowClient()

    # get top 3 model runs and log to MLflow
    experiment = client.get_experiment_by_name(OPTUNA_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=3,
        order_by=["metrics.validation_rmse ASC"]
    )
    for run in runs:
        train_best_model(train, valid, y_val, dv, params=run.data.params)

    # select the model with the lowest valid RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.validation_rmse ASC"]
    )[0]

    # register the best model
    mlflow.register_model(
        model_uri=f"runs:/{best_run.info.run_id}/model",
        name="ny-taxi-tripduration"
    )


if __name__ == '__main__':
    train()
