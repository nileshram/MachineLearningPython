"""

Date of Creation : 15 Aug 2019
Author : Nilesh Ramnarain

"""
import logging.config
import os
import json

#import graph library
from graph.graphlib import GraphLib
#import datamodel
from model.datamodel import DataModel
#import ml-classifiers
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
        #define datamodels here
        dax_data = DataModel(filename="dax.csv")
        es_50_data = DataModel(filename="eurostoxx.csv")
         
        dax_logit = LogisticalRegression()
        estoxx_logit = LogisticalRegression()
        
        #Run classifier
        dax_logit.run_classifier(dax_data)
        estoxx_logit.run_classifier(es_50_data)
#         support_vector_machine = SupportVectorMachine()
#         support_vector_machine.run_classifier(dax_data)
         
        g = GraphLib()
#         g.plot_multimodel_roc_curve(dax_data, es_50_data)
        g.plot_multimodel_confusion_matrix(dax_data, es_50_data, ["Positive Returns", "Negative Returns"], 
                                           "Confusion Matrix - Logistic Regression")
#         g.plot_roc_curve(dax_data)
#         g.plot_returns(data)
#         g.plot_confusion_matrix(dax_data.c_matrix, ["Positive Returns", "Negative Returns"],
#                                 "Confusion Matrix - Logistic Regression")

    except Exception as e:
        print(e)

