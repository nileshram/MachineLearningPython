'''
Created on 9 Nov 2019

@author: nilesh
'''
from abc import abstractmethod, ABCMeta
from sklearn import linear_model, svm, model_selection
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
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
        self._logger.info("Beginning test train split")
        data.x_train, data.x_test, data.y_train, data.y_test = model_selection.train_test_split(data.model[features], test_param,
                                                                            test_size=size, shuffle=False)
        self._logger.info("Finished test train data with split ratio: Training data {} Test data {} ".format(1 - size, size))

    def compute_confusion_matrix_test_sample(self, data, classifier):
        self.fit_model(data.x_train, data.y_train, classifier)
        y_pred_test = classifier.predict(data.x_test)
        self._logger.info("Confusion matrix as follows:")
        data.c_matrix_test = confusion_matrix(data.y_test, y_pred_test)
        self._logger.info(data.c_matrix_test)
        #pretty print confusion matrix with data labels
        c_matrix = self._pretty_print_confusion_matrix(data.y_test, y_pred_test)
        return c_matrix

    def compute_test_data_roc_metrics(self, data=None):
        data.fpr_test, data.tpr_test, data.thresholds_test = roc_curve(data.y_test, data.y_pred_test)
        data.roc_auc_score_test = roc_auc_score(data.y_test, data.y_pred_test)
        self._logger.info("ROC AUC Test Data Score: {}".format(data.roc_auc_score_test))

    def _pretty_print_confusion_matrix(self, y_test, y_pred):
        unique_label = np.unique([y_test, y_pred])
        c_matrix = pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=unique_label), 
            index=['true:{:}'.format(x) for x in unique_label], 
            columns=['pred:{:}'.format(x) for x in unique_label]
        )
        self._logger.info(c_matrix)
        return c_matrix

    def apply_gridsearch(self, data=None, classifier=None):
        self._grid = GridSearchCV(classifier, self._params, verbose=3)
        self._grid.fit(data.x_train, data.y_train)
        self._logger.info("Best params to use for chosen classifier: {}".format(self._grid.best_params_))
        grid_predictions = self._grid.predict(data.x_test)
        self._logger.info("Confusion Matrix as Follows: {}".format(confusion_matrix(data.y_test, grid_predictions)))
        self._logger.info("Classification Report: {}".format(classification_report(data.y_test, grid_predictions)))

    def get_predicted_probabilities(self, data=None, x_features=None, classifier=None):
        data.pred_prob = classifier.predict_proba(data.model[x_features])
        self._logger.info("Computed predicted probabilities")

    def _run_main_classifier(self, data=None, x_features=None, y_result=None, classifier=None):
        self._logger.info("Fitting model against specified parameters")
        self.fit_model(data.model[x_features], y_result, classifier)
        self.run_prediction(data, x_features, classifier)
        self.compute_confusion_matrix_main(data)
        self.compute_population_data_roc_metrics(data)
        self.get_predicted_probabilities(data=data, x_features=x_features, classifier=classifier)

    def _run_test_fit_classifier(self, data=None, x_features=None, y_result=None, classifier=None, size=None):
        self.train_test_split(data=data, features=x_features, size=size, test_param=y_result)
        self.fit_model(data.x_train, data.y_train, classifier)
        data.y_pred_test = classifier.predict(data.x_test)
        self.compute_test_data_roc_metrics(data=data)
    
class LogisticalRegression(Classification):
    
    def __init__(self):
        super(LogisticalRegression, self).__init__()
        self._init_classifier()
        self._init_grid_search_params()
        
    def _init_classifier(self):
        self.logreg_main = linear_model.LogisticRegression(penalty="l2", C = 100)
        self.logreg_test = linear_model.LogisticRegression(penalty="l2", C = 100)
    
    def _init_grid_search_params(self):
        self._params = {"C" : [100000, 10000, 1000, 100, 10, 1, .1, .001],
                        "penalty" : ["l1", "l2"]}
        
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
#         self.apply_gridsearch(data=data, classifier=linear_model.LogisticRegression())
        self._logger.info("Finished running through classifier")

    def compute_confusion_matrix_main(self, data):
        self._logger.info("Computing Confusion matrix for main data as follows:")
        data.c_matrix_main = confusion_matrix(data.model.log_return_sign, data.model.log_pred)
        self._logger.info(data.c_matrix_main)
        #pretty print confusion matrix with data labels
        c_matrix = self._pretty_print_confusion_matrix(data.model.log_return_sign, data.model.log_pred)
        return c_matrix

    def compute_population_data_roc_metrics(self, data):
        data.fpr_main, data.tpr_main, data.thresholds_main = roc_curve(data.model.log_return_sign, data.model.log_pred)
        data.roc_auc_score_main = roc_auc_score(data.model.log_return_sign, data.model.log_pred)
        self._logger.info("ROC AUC Main Data Score: {}".format(data.roc_auc_score_main))
        
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
        self._init_scaler_pipeline()
        self._init_classifier()
        self._init_grid_search_params()

    def _init_classifier(self):
        self.svm_main = Pipeline(self._steps)
        self.svm_test = Pipeline(self._steps)
        self._logger.info("Usable params in pipeline {}".format(self.svm_main.get_params().keys()))
    
    def _init_scaler_pipeline(self):
        self._steps = [('scaler', StandardScaler()), ('SVC', svm.SVC(C=100, probability=True))]

    def _init_grid_search_params(self):
        self._params = {"SVC__C" : [1000, 100, 10, 1, .1, .001]}
        
    def fit_model(self, x_param, y_param, classifier):
        classifier.fit(x_param, y_param)
        
    def run_prediction(self, data, x_features, classifier):
        self._logger.info("Running SVM Prediction")
        data.model["svm_pred"] = classifier.predict(data.model[x_features])
        self._logger.info("Actual Return Summary: {}".format(data.model.log_return_sign.value_counts()))
        self._logger.info("SVM Prediction Summary: {}".format(data.model.svm_pred.value_counts()))
    
    def run_classifier(self, data):
        self._logger.info("Running SVM Classifier for {}".format(data.filename))
        #Separate Data Parameters of x_features and y_response
        returns_sign = data.model.log_return_sign
        #"moving_average", "momentum", "sample_sigma"
        x_features = [header for header in list(data.model) for feat in ["lagged_return_1", "lagged_return_2", "moving_average", "momentum", "sample_sigma"] if feat in header]
        #Run main classifier
        self._run_main_classifier(data=data, x_features=x_features, 
                                  y_result=returns_sign, classifier=self.svm_main)
        #Apply train test split on classifier
        self._run_test_fit_classifier(data=data, x_features=x_features, 
                                      y_result=returns_sign, classifier=self.svm_test, size=0.25)
        #Perform optimal gridsearch for parameters here
#         self.apply_gridsearch(data=data, classifier=self.svm_test)
        #2D relationship model
        self._run_2D_visualisation(data=data, y_target=returns_sign, data_filter=["momentum_5d", "lagged_return_1"])
        self._logger.info("Finished running through classifier")
        
    def compute_confusion_matrix_main(self, data):
        self._logger.info("Computing Confusion matrix for main data as follows:")
        data.c_matrix_main = confusion_matrix(data.model.log_return_sign, data.model.svm_pred)
        self._logger.info(data.c_matrix_main)
        #pretty print confusion matrix with data labels
        c_matrix = self._pretty_print_confusion_matrix(data.model.log_return_sign, data.model.svm_pred)
        return c_matrix

    def compute_population_data_roc_metrics(self, data):
        data.fpr_main, data.tpr_main, data.thresholds_main = roc_curve(data.model.log_return_sign, data.model.svm_pred)
        data.roc_auc_score_main = roc_auc_score(data.model.log_return_sign, data.model.svm_pred)
        self._logger.info("ROC AUC Main Data Score: {}".format(data.roc_auc_score_main))
        
    #define the 2D relationship here
    def _run_2D_visualisation(self, data=None, y_target=None, data_filter=None):
        filtered_model = data.model[data_filter]
        svm_2d = svm.SVC(C=100, probability=True)
        svm_2d.fit(filtered_model, y_target)
        data.visual_model_2D = filtered_model
        data.SVM_model_2D = svm_2d

        
            
        
class ANN(Classification):
    
    def __init__(self):
        super(ANN, self).__init__()
        pass
    
    def run_classifier(self):
        pass
