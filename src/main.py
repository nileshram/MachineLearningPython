"""

Date of Creation : 15 Aug 2019
Author : Nilesh Ramnarain

"""
import logging.config
import os
import json

from graph.graphlib import GraphLib
from backtesting.pl_backtest import PLBacktestingEngine
from model.datamodel import DataModel
from ml.classifiers import LogisticalRegression, SupportVectorMachine, ANN

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
        
if __name__ == "__main__":
    ConfigurationFactory._configure_log()
    log = logging.getLogger("cqf_logger")
    log.info("Initialising Program For CQF Exam 3 Machine Learning with Python")
    try:
        #Create datamodel here
        dax_data = DataModel(filename="dax.csv", extended_features=True)
        estoxx_data = DataModel(filename="eurostoxx.csv", extended_features=True)
        
        #Create Logistic Regression Classifier Here
        dax_logit = LogisticalRegression()
        estoxx_logit = LogisticalRegression()
        #Create SVM Classifier Here
        dax_svm = SupportVectorMachine()
        estoxx_svm = SupportVectorMachine()
        #Create RNN Classifier here
        dax_ann = ANN()
        estoxx_ann = ANN()
        dax_ann.run_classifier(dax_data)
        estoxx_ann.run_classifier(estoxx_data)
        
        #Run classifier
        dax_svm.run_classifier(dax_data)
        estoxx_svm.run_classifier(estoxx_data)
        dax_logit.run_classifier(dax_data)
        estoxx_logit.run_classifier(estoxx_data)
        pl = PLBacktestingEngine()
        pl.run_backtest(data=dax_data, initial_capital=500000, bet_size=0.10, 
                        upper_bound=0.7, lower_bound=0.55)
        pl.run_backtest(data=estoxx_data, initial_capital=500000, bet_size=0.10, 
                        upper_bound=0.7, lower_bound=0.55)
        pl.compute_transitional_probabilities(dax_data)
        pl.compute_transitional_probabilities(estoxx_data)
  
        g = GraphLib()
        g.plot_multimodel_rnn_returns(dax_data, estoxx_data)
        g.plot_multimodel_rnn_prediction(dax_data, estoxx_data)
        g.plot_svm_2d_multimodel(dax_data, estoxx_data)
        g.plot_multimodel_pl_backtest(dax_data, estoxx_data)
        g.plot_transition_probabilities_multi_model(dax_data, estoxx_data)
        g.plot_multimodel_roc_curve(dax_data, estoxx_data)
        g.plot_multimodel_confusion_matrix(dax_data, estoxx_data, ["Positive Returns", "Negative Returns"], 
                                           "Confusion Matrix - LSTM")
    except Exception as e:
        print(e)

