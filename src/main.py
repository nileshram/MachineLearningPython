"""

Date of Creation : 15 Aug 2019
Author : Nilesh Ramnarain

"""
from builtins import staticmethod
from abc import abstractmethod, ABCMeta
from sklearn import linear_model, svm, model_selection
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
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
        self.lm = linear_model.LogisticRegression(C = 1e5)
    
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
        #norm_matrix = matrix * 1. / matrix.sum(axis=1)[:, np.newaxis] #standardised confusion matrix
        #plot heatmap here
        fig, ax = plt.subplots()
        cm = plt.cm.get_cmap('bwr')
        im = ax.imshow(matrix, cmap=cm)
        
        #add tick labels
        ticks = np.arange(len(target_names))
        plt.xticks(ticks, target_names[::-1], rotation=45)
        plt.yticks(ticks, target_names[::-1])
        
        # Loop over data dimensions and create text annotations.
        for i in range(len(target_names)):
            for j in range(len(target_names)):
                text = ax.text(j, i, matrix[i, j],
                               ha="center", va="center", color="black")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(title)
        fig.tight_layout()
        plt.colorbar(im)
        plt.show()

    def plot_roc_curve(self, data):
        fig, ax = plt.subplots()
        ax.plot(data.fpr_test, data.tpr_test, label="Test Data", color="purple")
        ax.plot(data.fpr_population, data.tpr_population, label="Population Data", color="blue")
        ax.plot([0,1],[0,1],'r--',label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        
        plt.title("ROC Curve for Predicted Returns Classifier")
        plt.xlabel("False positive rate (1 - Specificity)")
        plt.ylabel("True positive rate (Sensitivity)")
        #chart rendering
        plt.minorticks_on()
        ax.set_facecolor(color='whitesmoke')
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
    ConfigurationFactory._configure_log()
    log = logging.getLogger("cqf_logger")
    log.info("Initialising Program For CQF Exam 3 Machine Learning with Python")
    try:
        data = DataModel()
         
        logit = LogisticalRegression()
        logit.run_classifier(data)
         
        support_vector_machinem = SupportVectorMachine()
        support_vector_machinem.run_classifier(data)
         
        g = GraphLib()
        g.plot_roc_curve(data)
#         g.plot_returns(data)
#         g.plot_confusion_matrix(logit.c_matrix, ["Positive Returns", "Negative Returns"],
#                                 "Confusion Matrix - Logistic Regression")
         
    except Exception as e:
        print(e)

