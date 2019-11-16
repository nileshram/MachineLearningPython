'''
Created on 9 Nov 2019

@author: nilesh
'''
import numpy as np
from math import sqrt

class FeaturesEngineering:
    
    @staticmethod
    def compute_log_return(df, period=None):
        if period is None:
            period = 1
        df["log_return"] = np.log(df.Settle) - np.log(df.Settle.shift(period))
        df["log_return_sign"] = np.sign(df["log_return"])
        #add condition for binary classification if 0% then replace with negative return to adjust for margin
        df["log_return_sign"].replace(0, -1, inplace=True)
        return df
    
    @staticmethod
    def compute_lagged_returns(df, lag_number):
        for lag in range(1, lag_number+1):
            index = "lagged_return_{}".format(lag)
            df[index] = df.log_return.shift(lag)
        return df
    
    @staticmethod
    def compute_momentum_indicator(df, field=None, period=None):
        tag = "momentum_{}d".format(str(period))
        df[tag] = df[field] - df[field].shift(period)
        return df

    @staticmethod
    def compute_moving_average(df, field=None, period=None):
        tag = "moving_average_{}d".format(str(period))
        df[tag] = df[field].rolling(period).mean()
        return df
    
    @staticmethod
    def compute_periodic_standard_deviation(df, field=None, period=None):
        tag = "sample_sigma_{}".format(str(period))
        df[tag] = df[field].rolling(period).std()
        return df