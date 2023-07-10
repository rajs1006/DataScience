"""
    A data source can be single or multiple and this class or similar class can be used to 
    segregate the connection and Operation to different data sources.

    If any prebuilt feture store is used then this class can play a role fo connetor to 
    that feature store. 
"""

import pandas as pd


def sensorData(isTrainValidate: bool):
    if isTrainValidate:
        df = pd.read_csv("data/historical_sensor_data.csv", sep=",")
        X = df[["sensor_1", "sensor_2"]].values
        y = df[
            [
                "label",
            ]
        ].values.ravel()

        return X, y
    else:
        df = pd.read_csv("data/latest_sensor_data.csv", sep=",")
        return df.values
