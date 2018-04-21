import os
# import keras
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

    test_df = pd.read_csv(testDataFilePath, parse_dates=['TIME'], date_parser=dateParser)
    test_df.columns = ['terminalno',
            'time',
            'trip_id',
            'longitude',
            'latitude',
            'direction',
            'height',
            'speed',
            'callstate']
    feat_serv = FeatureEngineering(test_df)
    X, featureCols = feat_serv.create_X_dataset()
    X = feat_serv.normalization(X, featureCols)

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
    X, featureCols = featureService.create_X_dataset()
    Y = featureService.create_Y_dataset()
    X = featureService.normalization(X, featureCols)

    model = linear_model.LinearRegression()
    model.fit(X, Y)
    predictions = model_prediction(model)

    print('------- saving results -------')
    predictions.to_csv('./model/results.csv', index=False)

if __name__ == "__main__":

    print('-------- run begins --------')
    run()
