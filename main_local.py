import os
import pandas as pd
import numpy as np
from math import sqrt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from feature_engineering import FeatureEngineering

trainDataFilePath = os.path.join(os.getcwd(), 'local.csv')
dateParser = lambda x: pd.datetime.fromtimestamp(float(x))

def model_evaluation(model, X, Y):
    predict = model.predict(X)
    return mean_squared_error(Y, predict)

def run():

    df = pd.read_csv(trainDataFilePath, parse_dates=['TIME'], date_parser=dateParser)
    df.columns = ['terminalno',
            'time',
            'trip_id',
            'longitude',
            'latitude',
            'direction',
            'height',
            'speed',
            'callstate',
            'y']
    
    featureService = FeatureEngineering(df)
    X = featureService.create_X_dataset()
    Y = featureService.create_Y_dataset()
    X = featureService.normalization(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)
    trainRMSE = sqrt(model_evaluation(model, X_train, Y_train))
    testRMSE = sqrt(model_evaluation(model, X_test, Y_test))

    print(trainRMSE)
    print(testRMSE)

if __name__ == "__main__":

    print('------- run begins -------')
    run()
