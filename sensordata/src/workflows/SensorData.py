import os
from typing import Tuple

import numpy as np
import pandas as pd
from dagster import AssetOut, asset, job, multi_asset, op, repository
from data.LoadData import sensorData
from services.models.classification import Model
from utils.Logger import logger

log = logger(__file__)

"""
    A folder to store everything

    We could also use S3/Azure Blobs or even Database for this.

    A singleton connection objects can be created to Achieve this, 
    Something similar can be wriiten as per below.

    ----------------------  Sample Code to create sigleton DB connection -------------------
    class Connection:
        def __new__(cls, *args, **kwargs):
            if not hasattr(cls, "_instance"):
                cls._instance = object.__new__(cls, *args, **kwargs)
            return cls._instance

        def connect(self):
            if not hasattr(self, "_con"):
                log.warning("** Creating new instance of DB **")

                dbSSLDir = os.getenv("DB_SSL_DIR")
                user = os.getenv("DB_USER")
                self._con = psycopg2.connect(
                    dbname=os.getenv("DATABASE"),
                    user=user,
                    host=os.getenv("DB_HOST"),
                    port=os.getenv("DB_PORT"),
                    sslmode="verify-full",
                    sslcert=os.path.join(dbSSLDir, f"{user}.crt"),
                    sslkey=os.path.join(dbSSLDir, f"{user}.key"),
                    sslrootcert=os.path.join(dbSSLDir, "ca.crt"),
                )

            return self._con
"""
FOLDER = "experiments/sensordata/"
os.makedirs(FOLDER, exist_ok=True)

## NOTE : I have used Dagster as an Orchetration Tool for Train, Validate and Predict.
# Simple PYTHON methods could have worked as well, but in my experience simple python
# methods start to get complicated as number of models increase and also sometimes the
# input of one model depends on output of another model and so using a Orchestration tool
# is a better solution.
"""
    Dagster is Pipeline Orchestration tool, built to handle Data and ML Pipelines.
    There is a leraning curve in the begining and so this tool could be replaced
    by some other more use-case specific tools, like Airflow, Prefect or even ZenML.

    Dagster supports Scheduling and  manual triggering

    I have found a very good use case using ZenML and Feast.
    https://sourabhraj.net/2023/06/07/overview-building-an-efficient-and-scalable-mlops-workflow-with-feast-and-zenml/
"""


@multi_asset(
    outs={
        "feature_asset": AssetOut(),
        "taget_asset": AssetOut(),
    }
)
def getHistoricalData() -> Tuple[np.ndarray, np.ndarray]:
    features, target = sensorData(isTrainValidate=True)

    return features, target


@asset
def getInferenceData() -> np.ndarray:
    features = sensorData(isTrainValidate=False)

    return features


@op
def validate(features, target):
    model = Model()

    bestModel = {"best_score": 0}
    for modelName in ["LR", "GaussianProcess"]:
        model(modelName)
        bestParams, bestScore, cvResults = model.validate(features, target)

        log.info(f"Best Score for {modelName} is {bestScore}")
        log.debug(f"CV Score for {modelName} is {cvResults}")
        if bestScore > bestModel["best_score"]:
            bestModel.update(
                {"score": bestScore, "model": modelName, "params": bestParams}
            )

    Model.toPickle(cvResults, os.path.join(FOLDER, "cv_results.pkl"))
    Model.toPickle(bestModel, os.path.join(FOLDER, "best_parameters.pkl"))


@op
def train(features, target):
    model = Model()

    bestModel = model.fromPickle(os.path.join(FOLDER, "best_parameters.pkl"))
    log.info(f"Training with parameters {bestModel}")

    model(bestModel["model"])
    model.fit(features, target, bestModel["params"])
    Model.toPickle(model, os.path.join(FOLDER, "trained_model.pkl"))

    log.debug(f"Training : END")


@op
def predict(features):
    model = Model.fromPickle(os.path.join(FOLDER, "trained_model.pkl"))
    predictions = model.predict(features)

    Model.toPickle(predictions, os.path.join(FOLDER, "predictions.pkl"))

    log.debug(f"Prediction :  END")

#-------------------- Orchestration JOBs to run different Operations -------------------- 
@job(name="validate_sensordata")
def ValidateJob():
    log.debug(f"Validate :  START")

    features, target = getHistoricalData()
    validate(features, target)

    log.debug(f"Validate :  END")


@job(name="validate_train_sensordata")
def ValidateTrainJob():
    log.debug(f"Validate and Train :  START")

    features, target = getHistoricalData()
    validate(features, target)
    train(features, target)

    log.debug(f"Validateand Train  :  END")


@job(name="train_sensordata")
def TrainJob():
    log.debug(f"Validate :  START")

    features, target = getHistoricalData()
    train(features, target)

    log.debug(f"Validate :  END")


@job(name="predict_sensordata")
def PredictJob():
    log.debug(f"Validate :  START")

    features = getInferenceData()
    predict(features)

    log.debug(f"Validate :  END")


@repository
def hello_cereal_repository():
    return [ValidateJob, ValidateTrainJob, TrainJob, PredictJob]
