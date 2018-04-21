import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2

class FeatureEngineering(object):

    def __init__(self, df):

        df['long'] = np.radians(df.longitude)
        df['lat'] = np.radians(df.latitude)
        df['distance'] = self.haversine_np(
            df.long.shift(),
            df.lat.shift(),
            df.loc[1:, 'long'],
            df.loc[1:, 'lat'])
        df['dir_diff'] = df.direction.diff()
        df['height_diff'] = df.height.diff()
        df = df.fillna(0)

        self.df = df
    
    def create_X_dataset(self):

        intermediateFeatures = self.df.groupby(['terminalno', 'trip_id']).apply(self.map_features)
        intermediateFeatures['right_n_left_turns'] = np.add(intermediateFeatures.right_dir_sum, intermediateFeatures.left_dir_sum)
        features = intermediateFeatures.groupby('terminalno').apply(self.features_per_trip)
        features['total_height_changes_per_person'] = np.add(features.up_height_per_trip, features.down_height_per_trip)

        return features
    
    def create_Y_dataset(self):

        Y = self.df[['terminalno', 'y']].drop_duplicates()
        Y.index = Y.terminalno
        Y.pop('terminalno')
        
        return Y
    
    def normalization(self, dataframe):
        normalized = (dataframe - dataframe.mean()) / (dataframe.min())
        return normalized
    
    def map_features(self, dataframe):

        featuresMap = {
            'speed_mean': dataframe.speed.mean(),
            'distance_sum': dataframe.distance.sum(),
            'callstate_0_count': dataframe[dataframe.callstate == 0].callstate.count(),
            'callstate_1_count': dataframe[dataframe.callstate == 1].callstate.count(),
            'callstate_2_count': dataframe[dataframe.callstate == 2].callstate.count(),
            'callstate_3_count': dataframe[dataframe.callstate == 3].callstate.count(),
            'callstate_4_count': dataframe[dataframe.callstate == 4].callstate.count(),
            'up_height_sum': dataframe[dataframe.height_diff > 0].height_diff.count(),
            'down_height_sum': dataframe[dataframe.height_diff < 0].height_diff.count(),
            'height_change_sum': np.abs(dataframe.height_diff).sum(),
            'right_dir_sum': dataframe[dataframe.dir_diff > 0].dir_diff.count(),
            'left_dir_sum': dataframe[dataframe.dir_diff < 0].dir_diff.count()}
        
        return pd.Series(featuresMap, index=featuresMap.keys())
    
    def features_per_trip(self, dataframe):

        featuresMap = {
            'speed_per_trip': dataframe.speed_mean.mean(),
            'distance_per_trip': dataframe.distance_sum.mean(),
            'callstate_0_times_per_trip': dataframe.callstate_0_count.sum(),
            'callstate_1_times_per_trip': dataframe.callstate_1_count.sum(),
            'callstate_2_times_per_trip': dataframe.callstate_2_count.sum(),
            'callstate_3_times_per_trip': dataframe.callstate_3_count.sum(),
            'callstate_4_times_per_trip': dataframe.callstate_4_count.sum(),
            'height_change_per_trip': dataframe.height_change_sum.mean(),
            'up_height_per_trip': dataframe.up_height_sum.mean(),
            'down_height_per_trip': dataframe.down_height_sum.mean(),
            'right_turns_per_trip': dataframe.right_dir_sum.mean(),
            'left_turns_per_trip': dataframe.left_dir_sum.mean(),
            'total_turns_per_trip': dataframe.right_n_left_turns.sum()}
        
        return pd.Series(featuresMap, index=featuresMap.keys())
    
    def haversine_np(self, lon1, lat1, lon2, lat2):
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c
        return km


    
