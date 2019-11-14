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
        im1 = ax1.imshow(data_1.c_matrix_main, cmap=cm)

        #add tick labels
        ticks = np.arange(len(target_names))
        plt.xticks(ticks, target_names[::-1], rotation=45)
        plt.yticks(ticks, target_names[::-1])
        
        # Loop over data dimensions and create text annotations.
        for i in range(len(target_names)):
            for j in range(len(target_names)):
                text = ax1.text(j, i, data_1.c_matrix_main[i, j],
                               ha="center", va="center", color="black")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("".join([title, " - DAX"]))
        plt.colorbar(im1)
        
        plt.subplot(grid[0,1])
        ax2 = plt.gca()
        cm2 = plt.cm.get_cmap('bwr')
        im2 = ax2.imshow(data_2.c_matrix_main, cmap=cm2)
        
        #add tick labels
        ticks = np.arange(len(target_names))
        plt.xticks(ticks, target_names[::-1], rotation=45)
        plt.yticks(ticks, target_names[::-1])
        
        # Loop over data dimensions and create text annotations.
        for i in range(len(target_names)):
            for j in range(len(target_names)):
                text = ax2.text(j, i, data_2.c_matrix_main[i, j],
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
        ax1.plot(data_1.fpr_test, data_1.tpr_test, label="Test Data (AUC: {})".format(format(round(data_1.roc_auc_score_test, 4))), color="lime")
        ax1.plot(data_1.fpr_main, data_1.tpr_main, label="Population Data (AUC: {})".format(round(data_1.roc_auc_score_main, 4)), color="blue")
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
        ax2.plot(data_2.fpr_test, data_2.tpr_test, label="Test Data (AUC: {})".format(round(data_2.roc_auc_score_test, 4)), color="lime")
        ax2.plot(data_2.fpr_main, data_2.tpr_main, label="Population Data (AUC: {})".format(round(data_2.roc_auc_score_main, 4)), color="blue")
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
    
    def plot_transition_probabilities_multi_model(self, data_1, data_2):
        #define gridspec here
        grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.3)
        #Add graphs to gridspec dynamically
        plt.subplot(grid[0,0])
        #Duplicate axes here
        ax1 = plt.gca()
        ax1.scatter([x[0]for x in data_1.correct_prob], [x[1] for x in data_1.correct_prob], label="Correct Prediction: {}".format(len(data_1.correct_prob)), color="lime")
        ax1.scatter([x[0]for x in data_1.incorrect_prob], [x[1] for x in data_1.incorrect_prob], label="Incorrect Prediction: {}".format(len(data_1.incorrect_prob)), color="red")
        
        plt.title("Transition Probability Prediction - DAX")
        plt.xlabel("Returns Direction")
        plt.ylabel("Probabilities")
        #chart rendering
        plt.minorticks_on()
        ax1.set_facecolor(color='whitesmoke')
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.legend()


        plt.subplot(grid[0,1])
        ax2 = plt.gca()
        ax2.scatter([x[0]for x in data_2.correct_prob], [x[1] for x in data_2.correct_prob], label="Correct Prediction: {}".format(len(data_2.correct_prob)), color="lime")
        ax2.scatter([x[0]for x in data_2.incorrect_prob], [x[1] for x in data_2.incorrect_prob], label="Incorrect Prediction: {}".format(len(data_2.incorrect_prob)), color="red")
        
        plt.title("Transition Probability Prediction - ESTOXX")
        plt.xlabel("Returns Direction")
        plt.ylabel("Probabilities")
        #chart rendering
        plt.minorticks_on()
        ax2.set_facecolor(color='whitesmoke')
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.legend()
        #display plots
        plt.show()

    def plot_multimodel_pl_backtest(self, data_1, data_2):
        #define gridspec here
        grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.3)
        #Add graphs to gridspec dynamically
        plt.subplot(grid[0,0])
        #Duplicate axes here
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(data_1.model.Date, list(data_1.pl_cumsum)[1:], label="Cumulative sum P&L", color="blue")
        ax2.plot(data_1.model.Date, data_1.model.Settle, label="Daily Settle Price", color="mediumorchid")
        plt.title("P&L Backtesting Betting - DAX")
        plt.xlabel("Date")
        ax1.set_ylabel("Portfolio Value (EUR)")
        ax2.set_ylabel("Settlement Price")
        #chart rendering
        plt.minorticks_on()
        ax1.set_facecolor(color='whitesmoke')
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        # ask matplotlib for the plotted objects and their labels
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        
        plt.subplot(grid[0,1])
        ax3 = plt.gca()
        ax4 = ax3.twinx()
        ax3.plot(data_2.model.Date, list(data_2.pl_cumsum)[1:], label="Cumulative sum P&L", color="blue")
        ax4.plot(data_2.model.Date, data_2.model.Settle, label="Daily Settle Price", color="mediumorchid")
        plt.title("P&L Backtesting Betting - ESTOXX 50")
        plt.xlabel("Date")
        ax3.set_ylabel("Portfolio Value (EUR)")
        ax4.set_ylabel("Settlement Price")
        #chart rendering
        plt.minorticks_on()
        ax3.set_facecolor(color='whitesmoke')
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        # ask matplotlib for the plotted objects and their labels
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        ax4.legend(lines3 + lines4, labels3 + labels4, loc=0)
        #display plots
        plt.show()
    