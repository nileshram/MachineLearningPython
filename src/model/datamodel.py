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
    
    def __init__(self, filename=None):
        if filename is None:
            self.filename = "dax.csv"
        else:
            self.filename = filename
        self.model = DataManager.load_data("data", self.filename)
        self._convert_to_datetime()
        self.compute_model_features()
        self._clean_datamodel()
        
    def _clean_datamodel(self):
        self.model.dropna(inplace=True)
        self.model.sort_values(by='Date')
    
    def compute_model_features(self):
        FeaturesEngineering.compute_log_return(self.model)
        FeaturesEngineering.compute_lagged_returns(self.model, 5)
        FeaturesEngineering.compute_periodic_standard_deviation(self.model, field="log_return", period=10)
        FeaturesEngineering.compute_momentum_indicator(self.model, field="Settle",  period=5)
        FeaturesEngineering.compute_moving_average(self.model, field="Settle", period=20)
        
    def _convert_to_datetime(self):
        self.model.Date = pd.to_datetime(self.model.Date, format='%d/%m/%Y')