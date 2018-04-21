import os
import keras
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from feature_engineering import FeatureEngineering

trainDataFilePath = '/data/dm/train.csv'
testDataFilePath = '/data/dm/test.csv'

dateParser = lambda x: pd.datetime.fromtimestamp(float(x))

def model_evaluation(model, X, Y):

    predict = model.predict(X)
    return mean_squared_error(Y, predict)

def model_prediction(model):

    test_df = pd.read_csv(testDataFilePath, index_col=0, squeeze=True, parse_dates=['TIME'], date_parser=dateParser)
    feat_serv = FeatureEngineering(test_df)
    X = feat_serv.create_X_dataset()

    predict = model.predict(X)
    predict = pd.Series(predict[:,0])
    Id = pd.Series(X.index)
    predict = pd.DataFrame([Id, predict]).T
    predict.columns=['Id', 'Pred']

    return predict

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

    model = linear_model.LinearRegression()
    model.fit(X, Y)
    predictions = model_prediction(model)

    print('------- saving results -------')
    predictions.to_csv('./model/results.csv', index=False)

if __name__ = "__main__":

    print('-------- run begins --------')
    run()
