"""

Date of Creation : 15 Aug 2019
Author : Nilesh Ramnarain

"""
from builtins import staticmethod
from abc import abstractmethod, ABCMeta
from sklearn import linear_model, svm, model_selection
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import statsmodels.api as sm
import numpy as np
import pandas as pd
import logging.config
import os
import json

#import graph library
from graph.graphlib import GraphLib
#import datamodel
from model.datamodel import DataModel

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
    
class Classification(metaclass=ABCMeta):
    
    def __init__(self):
        self._logger = logging.getLogger("cqf_logger")
    
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
    
    @abstractmethod
    def train_test_split(self):
        raise NotImplementedError("Should implement method train_test_split()")
    
class LogisticalRegression(Classification):
    
    def __init__(self):
        super(LogisticalRegression, self).__init__()
        self._init_classifier()

    def _init_classifier(self):
        self.lm = linear_model.LogisticRegression(penalty="l2", C = 1e5)
    
    def fit_model(self, x_param, y_param):
        self.lm.fit(x_param, y_param)
        
    def run_classifier(self, data):
        self._logger.info("Running Logisitical Regression Classifier")
        #perform the regression on returns here
        returns_sign = data.model.log_return_sign
        lagged_headers = [header for header in list(data.model) if "lagged" in header]
        #fitting done here
        self._logger.info("Fitting model against specified parameters")
        self.fit_model(data.model[lagged_headers], returns_sign)
        self.run_prediction(data, lagged_headers)
        data.model["logistic_return"] = abs(data.model.log_return) * data.model.log_pred

        self._logger.info("Logistic Return Prediction Data Summary: {}".format(data.model.log_pred.value_counts()))
        self._logger.info("Logistic Regression data model: {}, with size {}".format(data.model.head(), len(data.model.log_pred)))
        self._logger.info("Logistic Regression score: {}".format(self.lm.score(data.model[lagged_headers], returns_sign)))
        
        self.train_test_split(data=data, features=lagged_headers, size=0.7, test_param=returns_sign)
        
        ###Test data
        #compute predicted probabilities here
        self.compute_test_data_predicted_probabilities(data)
        #compute metrics for ROC curve
        self.compute_test_data_roc_metrics(data)

        ###Population data
        #compute predicted probabilities here
        self.compute_population_data_predicted_probabilities(data, lagged_headers)
        #compute metrics for ROC curve
        self.compute_population_data_roc_metrics(data, data.model.log_pred)
        
    def run_prediction(self, data, x_features):
        self._logger.info("Running Logistical Regression Prediction")
        data.model["log_pred"] = self.lm.predict(data.model[x_features])
    
    def train_test_split(self, data=None, features=None, size=None, test_param=None):
        #default split to half the data set
        if size is None:
            size = 0.5
        data.x_train, data.x_test, data.y_train, data.y_test = model_selection.train_test_split(data.model[features], test_param,
                                                                            test_size=size, shuffle=False)
         
        self._logger.info("Running second logistic classifier on training set")
        logit2 = linear_model.LogisticRegression(C = 1e5)
        logit2.fit(data.x_train, data.y_train)
        data.y_pred = logit2.predict(data.x_test)
        print("Confusion matrix as follows:")
        data.c_matrix = confusion_matrix(data.y_test, data.y_pred)
        print(data.c_matrix)

        unique_label = np.unique([data.y_test, data.y_pred])
        cmtx = pd.DataFrame(
            confusion_matrix(data.y_test, data.y_pred, labels=unique_label), 
            index=['true:{:}'.format(x) for x in unique_label], 
            columns=['pred:{:}'.format(x) for x in unique_label]
        )
        print(cmtx)
        return cmtx
    
    def compute_test_data_predicted_probabilities(self, data):
        self._logger.info("Computing predicted probabilities on test set of data")
        data.y_pred_prob_test = self.lm.predict_proba(data.x_test)[:, 1]
        
    def compute_test_data_roc_metrics(self, data):
        data.fpr_test, data.tpr_test, data.thresholds_test = roc_curve(data.y_test, data.y_pred_prob_test)

    def compute_population_data_predicted_probabilities(self, data, features):
        self._logger.info("Computing predicted probabilities on population data")
        data.y_pred_prob_population = self.lm.predict_proba(data.model[features])[:, 1]
        
    def compute_population_data_roc_metrics(self, data, y_population):
        data.fpr_population, data.tpr_population, data.thresholds_population = roc_curve(y_population, data.y_pred_prob_population)
    
class SupportVectorMachine(Classification):
    
    def __init__(self):
        super(SupportVectorMachine, self).__init__()
        self._init_classifier()

    def _init_classifier(self):
        self.svm = svm.SVC(C = 1e5, probability=True)
    
    def fit_model(self, x_param, y_param):
        self.svm.fit(x_param, y_param)
        
    def run_classifier(self, data):
        self._logger.info("Running SVM Classifier")
        #perform the regression on returns here
        returns_sign = data.model.log_return_sign
        lagged_headers = [header for header in list(data.model) if "lagged" in header]
        #fitting done here
        self._logger.info("Fitting model against specified parameters")
        self.fit_model(data.model[lagged_headers], returns_sign)
        self.run_prediction(data, lagged_headers)
        data.model["svm_return"] = abs(data.model.log_return) * data.model.svm_pred
        print("SVM score: {}".format(self.svm.score(data.model[lagged_headers], returns_sign)))
        
        self.train_test_split(data=data, features=lagged_headers, size=0.7, test_param=returns_sign)
        
    def run_prediction(self, data, x_features):
        self._logger.info("Running SVM Prediction")
        data.model["svm_pred"] = self.svm.predict(data.model[x_features])

    def train_test_split(self, data=None, features=None, size=None, test_param=None):
        #default split to half the data set
        if size is None:
            size = 0.5
        self.x_train, self.x_test, self.y_train, self.y_test = model_selection.train_test_split(data.model[features], test_param,
                                                                            test_size=size, shuffle=False)
        print("Running second svm classifier on training set")
        svm2 = svm.SVC(C = 1e5, probability=True)
        svm2.fit(self.x_train, self.y_train)
        y_pred = svm2.predict(self.x_test)
        print("Confusion matrix as follows:")
        #first arguments to confusion matrix is true values and the second is predicted vals
        self.c_matrix = confusion_matrix(self.y_test, y_pred)
        print(self.c_matrix)

        unique_label = np.unique([self.y_test, y_pred])
        cmtx = pd.DataFrame(
            confusion_matrix(self.y_test, y_pred, labels=unique_label), 
            index=['true:{:}'.format(x) for x in unique_label], 
            columns=['pred:{:}'.format(x) for x in unique_label]
        )
        print(cmtx)
    
class ANN(Classification):
    
    def __init__(self):
        super(ANN, self).__init__()
        pass
    
    def run_classifier(self):
        pass

    def train_test_split(self, data=None, features=None, size=None, test_param=None):
        #default split to half the data set
        if size is None:
            size = 0.5
        self.x_train, self.x_test, self.y_train, self.y_test = model_selection.train_test_split(data.model[features], test_param,
                                                                            test_size=size, shuffle=False)
        
if __name__ == "__main__":
    ConfigurationFactory._configure_log()
    log = logging.getLogger("cqf_logger")
    log.info("Initialising Program For CQF Exam 3 Machine Learning with Python")
    try:
        #define datamodels here
        dax_data = DataModel(filename="dax.csv")
        es_50_data = DataModel(filename="eurostoxx.csv")
         
        dax_logit = LogisticalRegression()
        estoxx_logit = LogisticalRegression()
        
        #Run classifier
        dax_logit.run_classifier(dax_data)
        estoxx_logit.run_classifier(es_50_data)
#         support_vector_machinem = SupportVectorMachine()
#         support_vector_machinem.run_classifier(dax_data)
         
        g = GraphLib()
        g.plot_multimodel_confusion_matrix(dax_data, es_50_data, ["Positive Returns", "Negative Returns"], 
                                           "Confusion Matrix - Logistic Regression")
#         g.plot_roc_curve(dax_data)
#         g.plot_returns(data)
#         g.plot_confusion_matrix(dax_data.c_matrix, ["Positive Returns", "Negative Returns"],
#                                 "Confusion Matrix - Logistic Regression")
         
    except Exception as e:
        print(e)

