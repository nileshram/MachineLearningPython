'''
Created on 9 Nov 2019

@author: nilesh
'''
import matplotlib.pyplot as plt
import numpy as np

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
        
    def plot_multimodel_confusion_matrix(self, data_1, data_2, target_names, title):
        #define gridspec here
        grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.3)
        #Add graphs to gridspec dynamically
        plt.subplot(grid[0,0])
        #Duplicate axes here
        ax1 = plt.gca()
        cm = plt.cm.get_cmap('bwr')
        im1 = ax1.imshow(data_1.c_matrix, cmap=cm)

        #add tick labels
        ticks = np.arange(len(target_names))
        plt.xticks(ticks, target_names[::-1], rotation=45)
        plt.yticks(ticks, target_names[::-1])
        
        # Loop over data dimensions and create text annotations.
        for i in range(len(target_names)):
            for j in range(len(target_names)):
                text = ax1.text(j, i, data_1.c_matrix[i, j],
                               ha="center", va="center", color="black")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("".join([title, " - DAX"]))
        plt.colorbar(im1)
        
        plt.subplot(grid[0,1])
        ax2 = plt.gca()
        cm2 = plt.cm.get_cmap('bwr')
        im2 = ax2.imshow(data_2.c_matrix, cmap=cm2)
        
        #add tick labels
        ticks = np.arange(len(target_names))
        plt.xticks(ticks, target_names[::-1], rotation=45)
        plt.yticks(ticks, target_names[::-1])
        
        # Loop over data dimensions and create text annotations.
        for i in range(len(target_names)):
            for j in range(len(target_names)):
                text = ax2.text(j, i, data_2.c_matrix[i, j],
                               ha="center", va="center", color="black")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("".join([title, " - ESTOXX 50"]))
        plt.colorbar(im2)
        plt.show()

    def plot_multimodel_roc_curve(self, data_1, data_2):
        #define gridspec here
        grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.3)
        #Add graphs to gridspec dynamically
        plt.subplot(grid[0,0])
        #Duplicate axes here
        ax1 = plt.gca()
        ax1.plot(data_1.fpr_test, data_1.tpr_test, label="Test Data", color="purple")
        ax1.plot(data_1.fpr_population, data_1.tpr_population, label="Population Data", color="blue")
        ax1.plot([0,1],[0,1],'r--',label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        
        plt.title("ROC Curve for Predicted Returns Classifier - DAX")
        plt.xlabel("False positive rate (1 - Specificity)")
        plt.ylabel("True positive rate (Sensitivity)")
        #chart rendering
        plt.minorticks_on()
        ax1.set_facecolor(color='whitesmoke')
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.legend()


        plt.subplot(grid[0,1])
        ax2 = plt.gca()
        ax2.plot(data_2.fpr_test, data_2.tpr_test, label="Test Data", color="purple")
        ax2.plot(data_2.fpr_population, data_2.tpr_population, label="Population Data", color="blue")
        ax2.plot([0,1],[0,1],'r--',label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        
        plt.title("ROC Curve for Predicted Returns Classifier - ESTOXX 50")
        plt.xlabel("False positive rate (1 - Specificity)")
        plt.ylabel("True positive rate (Sensitivity)")
        #chart rendering
        plt.minorticks_on()
        ax2.set_facecolor(color='whitesmoke')
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.legend()
        #display plots
        plt.show()
        
    def plot_pl_backtest(self, data_1, data_2):
        pass
    