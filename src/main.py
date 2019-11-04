"""

Date of Creation : 15 Aug 2019
Author : Nilesh Ramnarain

"""
from builtins import staticmethod
from abc import abstractmethod, ABCMeta
from sklearn import linear_model, svm, model_selection
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging.config
import os
import json

class ConfigurationFactory:
    
    @staticmethod
    def create_config():
        conf = os.path.join("conf", "cqf_log.json")
        if os.path.exists(conf) and os.path.isfile(conf):
            with open(conf, "r") as f:
                config = json.load(f)
        else:
            log.info("Please check run configurations with python interpreter {PROJECT_LOC}/{PROJECT}")
        return config["model"]
    
    @staticmethod     
    def _configure_log():
        logconfjson = os.path.join("conf", "cqf_log.json")
        if os.path.exists(logconfjson) and os.path.isfile(logconfjson):
            with open(logconfjson, "r") as f:
                config = json.load(f)
            logging.config.dictConfig(config["log"])
        else:
            logging.basicConfig(level=logging.INFO)

class FeaturesEngineering:
    
    @staticmethod
    def compute_log_return(df, period=None):
        if period is None:
            period = 1
        df["log_return"] = np.log(df.Settle) - np.log(df.Settle.shift(period))
        return df
    
    @staticmethod
    def compute_lagged_returns(df, lag_number):
        for lag in range(1, lag_number+1):
            index = "lagged_return_{}".format(lag)
            df[index] = df.log_return.shift(lag)
        return df
    
    @staticmethod
    def compute_momentum_indicator(df):
        pass
    
    @staticmethod
    def compute_ewma(df):
        pass
    
class DataManager:
    
    @staticmethod
    def load_data(path, name):
        data_filepath = os.path.join(path, name)
        try:
            data = pd.read_csv(data_filepath)
        except Exception as e:
            log.info("Could not retrieve data object {}".format(e))
        return data
    
class DataModel:
    
    def __init__(self):
        self.model = DataManager.load_data("data", "dax.csv")
        self._convert_to_datetime()
        self.compute_model_features()
        self._clean_datamodel()
        
    def _clean_datamodel(self):
        self.model.dropna(inplace=True)
        self.model.sort_values(by='Date')
    
    def compute_model_features(self):
        FeaturesEngineering.compute_log_return(self.model)
        FeaturesEngineering.compute_lagged_returns(self.model, 5)
        
    def _convert_to_datetime(self):
        self.model.Date = pd.to_datetime(self.model.Date, format='%d/%m/%Y')
    
class SupervisedLearning(metaclass=ABCMeta):
    
    def __init__(self):
        pass
    
    @abstractmethod
    def _init_classifier(self):
        raise NotImplementedError("Should implement _init_classifier()")
    
    @abstractmethod
    def fit_model(self, x_param, y_param):
        pass
        
    @abstractmethod
    def run_classifier(self):
        raise NotImplementedError("Should implement method run_classifier()")
    
    @abstractmethod
    def run_prediction(self):
        raise NotImplementedError("Should implement method run_prediction()")
    
class LogisticalRegression(SupervisedLearning):
    
    def __init__(self):
        super(LogisticalRegression, self).__init__()
        self._init_classifier()

    def _init_classifier(self):
        self.lm = linear_model.LogisticRegression(C = 1e6)
    
    def fit_model(self, x_param, y_param):
        self.lm.fit(x_param, y_param)
        
    def run_classifier(self, data):
        log.info("Running Logisitical Regression Classifier")
        #perform the regression on returns here
        returns_sign = np.sign(data.model.log_return)
        lagged_headers = [header for header in list(data.model) if "lagged" in header]
        #fitting done here
        log.info("Fitting model against specified parameters")
        self.fit_model(data.model[lagged_headers], returns_sign)
        self.run_prediction(data.model[lagged_headers])
        data.model["logistic_return"] = data.model.log_return * data.model.log_pred
        print("Logistic Regression data model: {}, with size {}".format(data.model.head(), len(data.model.log_pred)))
        print("Logistic Regression score: {}".format(self.lm.score(data.model[lagged_headers], returns_sign)))
        print("Logistic Regression transition probabilities {}".format(self.lm.predict_proba(data.model[lagged_headers])))
        
        #Test on training set
        x_train, x_test, y_train, y_test = model_selection.train_test_split(data.model[lagged_headers], data.model.log_pred,
                                                                            test_size=0.7, shuffle=False)
        print("Training set of features: {} with size {}".format(x_train, len(x_train)))
        
        print("Running second logistic classifier on training set")
        logit2 = linear_model.LogisticRegression(C = 1e5)
        logit2.fit(x_train, y_train)
        y_pred = logit2.predict(x_test)
        self.c_matrix = confusion_matrix(y_test, y_pred)
        print(self.c_matrix)
        

    def run_prediction(self, model):
        log.info("Running Logistical Regression Prediction")
        data.model["log_pred"] = self.lm.predict(model)
    
class SupportVectorMachine(SupervisedLearning):
    
    def __init__(self):
        super(SupportVectorMachine, self).__init__()
        self._init_classifier()

    def _init_classifier(self):
        self.svm = svm.SVC(C = 1e6, probability=True)
    
    def fit_model(self, x_param, y_param):
        self.svm.fit(x_param, y_param)
        
    def run_classifier(self, data):
        log.info("Running SVM Classifier")
        #perform the regression on returns here
        returns_sign = np.sign(data.model.log_return)
        lagged_headers = [header for header in list(data.model) if "lagged" in header]
        #fitting done here
        log.info("Fitting model against specified parameters")
        self.fit_model(data.model[lagged_headers], returns_sign)
        self.run_prediction(data.model[lagged_headers])
        data.model["svm_return"] = data.model.log_return * data.model.svm_pred
        print("SVM score: {}".format(self.svm.score(data.model[lagged_headers], returns_sign)))
        
    def run_prediction(self, model):
        log.info("Running SVM Prediction")
        data.model["svm_pred"] = self.svm.predict(model)
    
class ANN(SupervisedLearning):
    
    def __init__(self):
        super(ANN, self).__init__()
        pass
    
    def run_classifier(self):
        pass

class GraphLib:
    
    def __init__(self):
        pass
    
    def plot_returns(self, data):
        ax1 = plt.gca()
        ax1.plot(data.model.Date, data.model.log_return.cumsum().apply(np.exp),
                 color="blue")
        ax1.plot(data.model.Date, data.model.logistic_return.cumsum().apply(np.exp),
                 color="lime")
        ax1.plot(data.model.Date, data.model.svm_return.cumsum().apply(np.exp),
                 color="purple")
        
        plt.title("Machine Learning - SVM/Logit Prediction of price returns")
        plt.minorticks_on()
        ax1.set_facecolor(color='whitesmoke')
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.legend()
        plt.show()
    
    def plot_confusion_matrix(self, matrix, target_names, title):
        norm_matrix = matrix * 1. / matrix.sum(axis=1)[:, np.newaxis] #standardised confusion matrix
        plt.imshow(norm_matrix, interpolation='nearest', cmap='bwr')
        plt.colorbar()
        ticks = np.arange(len(target_names))
        plt.xticks(ticks, target_names, rotation=45)
        plt.yticks(ticks, target_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(title)
        plt.tight_layout()
        plt.show()
        
if __name__ == "__main__":
    ConfigurationFactory._configure_log()
    log = logging.getLogger("cqf_logger")
    log.info("Initialising Program For CQF Exam 3 Machine Learning with Python")
    try:
        data = DataModel()
        
        logit = LogisticalRegression()
        logit.run_classifier(data)
        
#         svm = SupportVectorMachine()
#         svm.run_classifier(data)
        
        g = GraphLib()
#         g.plot_returns(data)
        g.plot_confusion_matrix(logit.c_matrix, ["Positive Returns", "Negative Returns"],
                                "Confusion Matrix - Logistic Regression")
        
    except Exception as e:
        print(e)

