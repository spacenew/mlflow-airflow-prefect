import pickle
import os

import pandas as pd

import optuna
from optuna.samplers import TPESampler

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error

import xgboost as xgb

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient


from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

RS = 2022

OPTUNA_EXPERIMENT_NAME = "xgboost-optuna"
mlflow.set_tracking_uri("sqlite:///mlflow-locally.db")
mlflow.set_experiment(OPTUNA_EXPERIMENT_NAME)
mlflow.xgboost.autolog()

@task
def process_data(filename):
    df = pd.read_parquet(filename)

    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])

    df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    df['duration'] = df['duration'].apply(lambda td: td.total_seconds() / 60)

    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


@task
def add_features(data_train, data_val):

    data_train['PU_DO'] = data_train['PULocationID'] + '_' \
                          + data_train['DOLocationID']
    data_val['PU_DO'] = data_val['PULocationID'] + '_' \
                        + data_val['DOLocationID']

    return data_train, data_val


@task
def split_vect_data_(data_train, data_val):

    categorical = ['PU_DO']
    numerical = ['trip_distance']

    dictvec = DictVectorizer()

    train_dicts = data_train[categorical + numerical].to_dict(orient='records')
    X_train = dictvec.fit_transform(train_dicts)

    val_dicts = data_val[categorical + numerical].to_dict(orient='records')
    X_val = dictvec.transform(val_dicts)

    target = 'duration'
    y_train = data_train[target].values
    y_val = data_val[target].values

    return X_train, X_val, y_train, y_val, dictvec

@task
def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

@task
def objective(trial, train, valid, y_val):

    params = {
        'max_depth': trial.suggest_int('max_depth', 4, 50, 1),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'objective': 'reg:squarederror',
        'eval_metric': trial.suggest_categorical('eval_metric', ['rmse']),
        'seed': RS
    }

    with mlflow.start_run():
        # mlflow.set_tag("model", "xgboost")
        # mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=15
        )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        # mlflow.log_metric("rmse", rmse)

    return {'loss': rmse}


@flow(task_runner=SequentialTaskRunner())
def main(train_path: str="./data/green_tripdata_2021-01.parquet",
         val_path:   str="./data/green_tripdata_2021-02.parquet",
         dest_path: str="./preprocess_data"):

    #Start tracking exp


    # EXPERIMENT_NAME = "xgboost-best-models"
    #
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # mlflow.set_experiment(EXPERIMENT_NAME)


    # Read and preprocess data
    X_train = process_data(train_path)
    X_val = process_data(val_path)

    # Add features
    X_train, X_val = add_features(X_train, X_val)

    #Vectorize and Split data
    X_train, X_val, y_train, y_val, dictvec = split_vect_data_(X_train, X_val)

    # create dest_path folder
    os.makedirs(dest_path, exist_ok=True)

    # Prepare data for xgboost
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    # save datasets
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "test.pkl"))
    dump_pickle((dictvec), os.path.join(dest_path, "dv.pkl"))

    # #Search best params
    sampler = TPESampler(seed=RS)
    study = optuna.create_study(direction='minimize',
                                    sampler=sampler)
    study.optimize(
        lambda trial: objective(
                            trial,
                            train,
                            valid,
                            y_val),
        n_trials=5
                   )


if __name__ == '__main__':
    main()

# prefect deployment build ./prefect_flow1.py:main --name "Example Deployment"
# prefect deployment apply ./main-deployment.yaml
# prefect work-queue preview c10f72af-76a4-4290-a4ae-d77505581a45
# prefect agent start c10f72af-76a4-4290-a4ae-d77505581a45


# prefect deployment build ./development.py:main --name "Example Deployment"