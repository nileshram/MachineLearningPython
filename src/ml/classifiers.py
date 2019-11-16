'''
Created on 9 Nov 2019

@author: nilesh
'''
from abc import abstractmethod, ABCMeta
from sklearn import linear_model, svm, model_selection
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
import logging
import numpy as np
import pandas as pd

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
    
    def train_test_split(self, data=None, features=None, size=None, test_param=None):
        #default split to half the data set
        if size is None:
            size = 0.5
        data.x_train, data.x_test, data.y_train, data.y_test = model_selection.train_test_split(data.model[features], test_param,
                                                                            test_size=size, shuffle=False)
         
        self._logger.info("Finished test train data with split ratio: Training data {} Test data {} ".format(1 - size, size))

    def _pretty_print_confusion_matrix(self, y_test, y_pred):
        unique_label = np.unique([y_test, y_pred])
        c_matrix = pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=unique_label), 
            index=['true:{:}'.format(x) for x in unique_label], 
            columns=['pred:{:}'.format(x) for x in unique_label]
        )
        self._logger.info(c_matrix)
        return c_matrix
    
class LogisticalRegression(Classification):
    
    def __init__(self):
        super(LogisticalRegression, self).__init__()
        self._init_classifier()

    def _init_classifier(self):
        self.logreg_main = linear_model.LogisticRegression(penalty="l2", C = 100)
        self.logreg_test = linear_model.LogisticRegression(penalty="l2", C = 100)
    
    def fit_model(self, x_param, y_param, classifier):
        classifier.fit(x_param, y_param)

    def run_prediction(self, data, x_features, classifier):
        self._logger.info("Running Logistical Regression Prediction")
        data.model["log_pred"] = classifier.predict(data.model[x_features])
        self._logger.info("Actual Return Summary: {}".format(data.model.log_return_sign.value_counts()))
        self._logger.info("Logistical Regression Prediction Summary: {}".format(data.model.log_pred.value_counts()))
        
    def run_classifier(self, data):
        self._logger.info("Running Logisitical Regression Classifier for {}".format(data.filename))
        #Separate Data Parameters of x_features and y_response
        returns_sign = data.model.log_return_sign
        lagged_headers = [header for header in list(data.model) if "lagged" in header]
        
        #Run main classifier
        self._run_main_classifier(data=data, x_features=lagged_headers, 
                                  y_result=returns_sign, classifier=self.logreg_main)
        
        #Apply train test split on classifier
        self._run_test_fit_classifier(data=data, x_features=lagged_headers, 
                                      y_result=returns_sign, classifier=self.logreg_test, size=0.25)
        
        #Perform optimal gridsearch for parameters here
#         self.apply_gridsearch(data)
#         
        self.get_predicted_probabilities(data, lagged_headers, self.logreg_main)
#         
        self._logger.info("Finished running through classifier")

    def _run_main_classifier(self, data=None, x_features=None, y_result=None, classifier=None):
        self._logger.info("Fitting model against specified parameters")
        self.fit_model(data.model[x_features], y_result, classifier)
        self.run_prediction(data, x_features, classifier)
        self.compute_confusion_matrix_main(data)
        self.compute_population_data_roc_metrics(data)
        self.compute_population_data_predicted_probabilities(data, x_features, classifier)
    
    def _run_test_fit_classifier(self, data=None, x_features=None, y_result=None, classifier=None, size=None):
        self.train_test_split(data=data, features=x_features, size=size, test_param=y_result)
        self.fit_model(data.x_train, data.y_train, classifier)
        data.y_pred_test = classifier.predict(data.x_test)
        self.compute_test_data_roc_metrics(data)
        self.compute_test_data_predicted_probabilities(data, classifier)
        
    def compute_confusion_matrix_test_sample(self, data, classifier):
        self.fit_model(data.x_train, data.y_train, classifier)
        y_pred_test = classifier.predict(data.x_test)
        self._logger.info("Confusion matrix as follows:")
        data.c_matrix_test = confusion_matrix(data.y_test, y_pred_test)
        self._logger.info(data.c_matrix_test)
        
        #pretty print confusion matrix with data labels
        c_matrix = self._pretty_print_confusion_matrix(data.y_test, y_pred_test)
        return c_matrix

    def compute_confusion_matrix_main(self, data):
        self._logger.info("Computing Confusion matrix for main data as follows:")
        data.c_matrix_main = confusion_matrix(data.model.log_return_sign, data.model.log_pred)
        self._logger.info(data.c_matrix_main)
        
        #pretty print confusion matrix with data labels
        c_matrix = self._pretty_print_confusion_matrix(data.model.log_return_sign, data.model.log_pred)
        return c_matrix

    def compute_test_data_roc_metrics(self, data):
        data.fpr_test, data.tpr_test, data.thresholds_test = roc_curve(data.y_test, data.y_pred_test)
        data.roc_auc_score_test = roc_auc_score(data.y_test, data.y_pred_test)
        self._logger.info("ROC AUC Test Data Score: {}".format(data.roc_auc_score_test))

    def compute_population_data_roc_metrics(self, data):
        data.fpr_main, data.tpr_main, data.thresholds_main = roc_curve(data.model.log_return_sign, data.model.log_pred)
        data.roc_auc_score_main = roc_auc_score(data.model.log_return_sign, data.model.log_pred)
        self._logger.info("ROC AUC Main Data Score: {}".format(data.roc_auc_score_main))
    
    def compute_test_data_predicted_probabilities(self, data, classifier):
        self._logger.info("Computing predicted probabilities on test set of data")
        data.y_pred_prob_test = classifier.predict_proba(data.x_test)[:, 1]

    def compute_population_data_predicted_probabilities(self, data, features, classifier):
        self._logger.info("Computing predicted probabilities on population data")
        data.y_pred_prob_population = classifier.predict_proba(data.model[features])[:, 1]
        
    def get_predicted_probabilities(self, data, x_features, classifier):
        data.pred_prob = classifier.predict_proba(data.model[x_features])
        self._logger.info("Computed predicted probabilities")
    
    def apply_gridsearch(self, data):
        self._params = {"C" : [100000, 10000, 1000, 100, 10, 1, .1, .001],
                        "penalty" : ["l1", "l2"]}
        self._grid = GridSearchCV(linear_model.LogisticRegression(),
                                  self._params, verbose=3)
        self._grid.fit(data.x_train, data.y_train)
        self._logger.info("Best params to use for chosen classifier: {}".format(self._grid.best_params_))
        grid_predictions = self._grid.predict(data.x_test)
        self._logger.info("Confusion Matrix as Follows: {}".format(confusion_matrix(data.y_test, grid_predictions)))
        self._logger.info("Classification Report: {}".format(classification_report(data.y_test, grid_predictions)))

    def test_regularisation_strengths(self, x_train, y_train, x_test, y_test):
        strengths = [100000, 10000, 1000, 100, 10, 1, .1, .001]
        #Create a scaler object
        sc = StandardScaler()
        #Fit the scaler to the training data and transform
        x_train_std = sc.fit_transform(x_train)
        #Apply the scaler to the test data
        x_test_std = sc.transform(x_test)
        
        for c in strengths:
            clf = linear_model.LogisticRegression(penalty='l1', C=c) #solver='liblinear'
            clf.fit(x_train, y_train)
            self._logger.info('C: {}'.format(c))
            self._logger.info('Coefficient of each feature: {}'.format(clf.coef_))
            self._logger.info('Training accuracy: {}'.format(clf.score(x_train_std, y_train)))
            self._logger.info('Test accuracy: {}'.format(clf.score(x_test_std, y_test)))
    
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
        pred_prob = self.svm.predict_proba(data.model[lagged_headers])
        print("max min transition probabilities :")
        print(max(pred_prob[:,1]))
        print(min(pred_prob[:,1]))
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