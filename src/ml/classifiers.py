'''
Created on 9 Nov 2019

@author: nilesh
'''
from abc import abstractmethod, ABCMeta
from sklearn import linear_model, svm, model_selection
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.grid_search import GridSearchCV
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import logging
import numpy as np
import pandas as pd

class Classification(metaclass=ABCMeta):
    '''
    Class Docs:
    This is the main class for classification that is used in scikit-learn; there are initial methods here
    in the abstract base class for which each of the additional classifiers can inherit from for further analysis
    which is specific to each classifier
    '''
    
    def __init__(self):
        self._logger = logging.getLogger("cqf_logger")
        self._logger.propagate = False
    
    @abstractmethod
    def _init_classifier(self):
        raise NotImplementedError("Should implement _init_classifier()")
    
    @abstractmethod
    def fit_model(self, x_param, y_param):
        raise NotImplementedError("Should implement fit_model()")
        
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
        self._steps = [('scaler', StandardScaler()), ('SVC', svm.SVC(C=1, probability=True))]

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
#         self._run_2D_visualisation(data=data, y_target=returns_sign, data_filter=["momentum_5d", "lagged_return_1"])
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
        svm_2d = svm.SVC(C=1000, probability=True)
        svm_2d.fit(filtered_model, y_target)
        data.visual_model_2D = filtered_model
        data.SVM_model_2D = svm_2d

class ANN(Classification):
    '''
    Class Docs:
    This is the main class for running our LSTM classifier; utilising TensorFlow 2. We have initialised 4 layers for use of analysis
    and have adjusted the dropout amounts and activation functions accordingly. All results are appended to the data model object for further plotting
    '''
    
    def __init__(self):
        super(ANN, self).__init__()
        self._init_classifier()
        self._init_scaler()
    
    def _init_classifier(self):
        self._regressor = Sequential()
    
    def _init_scaler(self):
        self._scaler = MinMaxScaler(feature_range=(0,1))
        self._scaler_target = MinMaxScaler(feature_range=(0,1))
        
    def run_classifier(self, data=None):
        #Initialise features and y_response here 
        x_features = ["Open", "High", "Low", "Settle", "sample_sigma_10d", "moving_average_20d", "momentum_5d", "RSI_14d", "stoch_k", "macd"]
#         x_features = ["sample_sigma_10d", "moving_average_20d", "momentum_5d", "RSI_14d", "stoch_k", "macd"]
        #Apply test train split here by date
        self._test_train_split(data=data, date_filter="2018-01-01")
        #Run scaling of parameters
        self._apply_train_data_scaling(data=data, features=x_features)
        #Prepare data set
        self._prepare_train_dataset(data=data, time_steps=60)
        #Run scaling of parameters for test dataset
        self._apply_test_data_scaling(data=data, features=x_features)
        #Apply scaling of test data set
        self._prepare_test_dataset(data=data, time_steps=60)
        #Run LSTM
        self._init_LSTM(data=data)
    
    def _test_train_split(self, data=None, date_filter=None):
        #Apply date filter to data model
        self._logger.info("Applying date filter range to datamodel for test train split")
        data.train_dataframe = data.model[data.model.Date < date_filter]
        data.test_dataframe = data.model[data.model.Date >= date_filter]
        data.train_target = data.train_dataframe.Settle
        
    def _apply_train_data_scaling(self, data=None, features=None):
        self._logger.info("Applying scaling to train data")
        data.scaled_train_target = self._scaler_target.fit_transform(data.train_target.values.reshape(-1, 1))
#         data.scaled_train_target = list(np.where(data.train_target.values == -1, 0, 1))
        self._logger.info("Applying MinMaxScaler() to training data")
        data.scaled_train_data = self._scaler.fit_transform(data.train_dataframe[features])
    
    def _apply_test_data_scaling(self, data=None, features=None):
        self._logger.info("Applying scaling to test data")
        data.scaled_test_target = self._scaler_target.fit_transform(data.test_dataframe.Settle.values.reshape(-1,1))
        self._logger.info("Applying MinMaxScaler() to test data")
        data.scaled_test_data = self._scaler.transform(data.test_dataframe[features])
        
    def _prepare_train_dataset(self, data=None, time_steps=None):
        data.time_steps = time_steps
        x_train = []
        y_train = []
        for t in range(time_steps, len(data.scaled_train_data)):
            x_train.append(data.scaled_train_data[t-time_steps:t])
            y_train.append(data.scaled_train_target[t])
        x_train, y_train = np.array(x_train), np.array(y_train)
        data.x_train = x_train
        data.y_train = y_train
        #prepare test set here
        data.previous_timestep_data = data.train_dataframe.tail(time_steps)
        #we now append this previous data onto the test set
        data.test_dataframe = data.previous_timestep_data.append(data.test_dataframe, ignore_index=True)

    def _prepare_test_dataset(self, data=None, time_steps=None):
        x_test = []
        y_test = []
        for t in range(time_steps, len(data.scaled_test_data)):
            x_test.append(data.scaled_test_data[t-time_steps:t])
            y_test.append(data.scaled_test_target[t])
        x_test, y_test = np.array(x_test), np.array(y_test)
        data.x_test = x_test
        data.y_test = y_test
        
        self._logger.info("x train data shape: {}".format(data.x_train.shape))
        self._logger.info("y train data shape: {}".format(data.y_train.shape))
        self._logger.info("x test data shape: {}".format(data.x_test.shape))
        
    
    def fit_model(self):
        pass
    
    def _init_LSTM(self, data=None):
        self._logger.info("Adding LSTM to regressor")
        self._regressor.add(LSTM(units=60, activation='relu', return_sequences=True, input_shape=(data.x_train.shape[1], data.x_train.shape[2])))
        self._regressor.add(Dropout(0.2))
        self._regressor.add(LSTM(units=60, activation='relu', return_sequences=True))
        self._regressor.add(Dropout(0.3))
        self._regressor.add(LSTM(units=80, activation='relu', return_sequences=True))
        self._regressor.add(Dropout(0.4))
        self._regressor.add(LSTM(units=120, activation='relu'))
        self._regressor.add(Dropout(0.5))
        #this is the output layer
        self._regressor.add(Dense(units=1, activation="sigmoid"))
        
        self._logger.info("TensorFlow Summary\n {}".format(self._regressor.summary()))
        #run regressor
        self._regressor.compile(optimizer='adam', loss="binary_crossentropy") 
        self._regressor.fit(data.x_train, data.y_train, epochs=1, batch_size=32)
        
        #get epoch loss here
#         data.history = self._regressor.fit(data.x_train, data.y_train, validation_split = 0.01, epochs=50, batch_size=100)
        
        data.y_pred_scaled = self._regressor.predict(data.x_test)
        #inverse transform
        data.y_pred = self._scaler_target.inverse_transform(data.y_pred_scaled)
        data.y_test = self._scaler_target.inverse_transform(data.y_test)
         
        self._compute_return_metrics(data=data, period=10)
        self._logger.info("Finished running LSTM regressor")
        
    def run_prediction(self):
        pass
    
    def _compute_return_metrics(self, data=None, period=None):
        data.lstm_period = period
        data.lstm_model = pd.DataFrame({"lstm_pred_settle" : data.y_pred.flatten(),
                                        "actual_settle" : data.y_test.flatten()})
        data.lstm_model["actual_log_return"] = np.log(data.lstm_model.actual_settle) - np.log(data.lstm_model.actual_settle.shift(period))
#         data.lstm_model["actual_log_return_sign"] = np.sign(data.lstm_model.actual_log_return)
#         data.lstm_model["actual_log_return_sign"].replace(0, -1, inplace=True)
        #compute predicted log return sign
        data.lstm_model["pred_log_return"] = np.log(data.lstm_model.lstm_pred_settle) - np.log(data.lstm_model.lstm_pred_settle.shift(period))
#         data.lstm_model["pred_log_return_sign"] = np.sign(data.lstm_model.pred_log_return)
#         data.lstm_model["pred_log_return_sign"].replace(0, -1, inplace=True)
        #drop all unused values
        data.lstm_model.dropna(inplace=True)
        
        
