import pandas as pd


def sensorData(isTrainValidate : bool):
    if isTrainValidate:
        df =  pd.read_csv("data/historical_sensor_data.csv",sep=',')
        X = df[['sensor_1', 'sensor_2']].values
        y = df[['label',]].values.ravel()

        return X, y
    else:
        df = pd.read_csv("data/latest_sensor_data.csv",sep=',')
        return df.values
