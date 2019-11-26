'''
Created on 9 Nov 2019

@author: nilesh
'''
import pandas as pd
import numpy as np
import os
import json
from model.features import FeaturesEngineering

class DataManager:
    
    @staticmethod
    def load_data(path, name):
        data_filepath = os.path.join(path, name)
        try:
            data = pd.read_csv(data_filepath)
        except Exception as e:
            print("Could not retrieve data object {}".format(e))
        return data
    
class DataModel:
    
    def __init__(self, filename=None, extended_features=None):
        if filename is None:
            self.filename = "dax.csv"
        else:
            self.filename = filename
        self.model = DataManager.load_data("data", self.filename)
        self._convert_to_datetime()
        self.compute_model_features(extended_features=extended_features)
        self._apply_features_shift()
        self._clean_datamodel()
        
    def _clean_datamodel(self):
        self.model.dropna(inplace=True)
        self.model.sort_values(by='Date')
        
    def _apply_features_shift(self, extended_features=None):
        if extended_features is False:
            self.model.sample_sigma_10d = self.model.sample_sigma_10d.shift()
            self.model.moving_average_20d = self.model.moving_average_20d.shift()
            self.model.momentum_5d = self.model.momentum_5d.shift()
        else:
        #shift new features
            self.model.sample_sigma_10d = self.model.sample_sigma_10d.shift()
            self.model.moving_average_20d = self.model.moving_average_20d.shift()
            self.model.momentum_5d = self.model.momentum_5d.shift()
            self.model.RSI_14d = self.model.RSI_14d.shift()
            self.model.stoch_k = self.model.stoch_k.shift()
            self.model.macd = self.model.macd.shift()
            
    def compute_model_features(self, extended_features=None):
        if extended_features is False:
            FeaturesEngineering.compute_log_return(self.model)
            FeaturesEngineering.compute_lagged_returns(self.model, 5)
            FeaturesEngineering.compute_periodic_standard_deviation(self.model, field="log_return", period=10)
            FeaturesEngineering.compute_momentum_indicator(self.model, field="Settle",  period=5)
            FeaturesEngineering.compute_moving_average(self.model, field="Settle", period=20)
        else:
            FeaturesEngineering.compute_log_return(self.model)
            FeaturesEngineering.compute_momentum_indicator(self.model, field="Settle",  period=5)
            FeaturesEngineering.compute_moving_average(self.model, field="Settle", period=20)
            FeaturesEngineering.compute_periodic_standard_deviation(self.model, field="log_return", period=10)
            #Add TA lib technical indicators
            FeaturesEngineering.compute_rsi(self.model, field="Settle", period=14)
            FeaturesEngineering.compute_macd(self.model, field="Settle", fast_period=12, slow_period=26, signal_period=9)
            FeaturesEngineering.compute_stochastic_k(self.model, slow_k_period=14, slow_d_period=3)
            
    def _convert_to_datetime(self):
        self.model.Date = pd.to_datetime(self.model.Date, format='%d/%m/%Y')