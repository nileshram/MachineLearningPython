'''
Created on 9 Nov 2019

@author: nilesh
'''
import numpy as np
from math import sqrt
from talib import RSI, MACD, STOCH

class FeaturesEngineering:
    '''
    Class Docs:
    This class is responsible for all of the features engineering that takes place in the project
    The methods here are static and are generic to be applied to any pandas dataframe with the relevant
    attributes specified by the user. The well known libray ta-lib has also been used for advanced 
    technical indicators
    '''
    
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
        tag = "sample_sigma_{}d".format(str(period))
        df[tag] = df[field].rolling(period).std()
        return df

    @staticmethod
    def compute_rsi(df, field=None, period=None):
        tag = "RSI_{}d".format(str(period))
        df[tag] = RSI(df[field].values.astype(float), timeperiod=period) 
        return df

    @staticmethod
    def compute_macd(df, field=None, fast_period=None, slow_period=None, signal_period=None):
        df['macd'], df['macd_signal'], df['macd_hist'] = MACD(df[field].values.astype(float), fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
        return df

    @staticmethod
    def compute_stochastic_k(df, slow_k_period=None, slow_d_period=None):
        df['stoch_k'], df['stoch_d'] = STOCH(df.High.values.astype(float), df.Low.values.astype(float), df.Settle.values.astype(float), slowk_period=float(slow_k_period), slowd_period=float(slow_d_period))
        return df

    @staticmethod
    def apply_shift_to_field(df, field=None, shift_size=None):
        tag = "{}_{}".format(field, shift_size)
        df[tag] = df[field].shift(shift_size)
        return df

#instead of .values may have to use as_matrix() mathod
    