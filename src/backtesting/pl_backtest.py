'''
Created on 11 Nov 2019

@author: nilesh
'''
import numpy as np

class PLBacktesting:
    
    @staticmethod
    def compute_transitional_probabilities(data):
        #prepare data here:
        actual_returns_sign = list(data.model.log_return_sign)
        pred_return_sign = list(data.model.log_pred)
        positive_return_prob = list(data.pred_prob[:, 1])
        negative_return_prob = list(data.pred_prob[:, 0])
        correct_prob = []
        incorrect_prob = []
        #now to get plot co-ordinates
        for x in range(len(actual_returns_sign)):
            if pred_return_sign[x] == actual_returns_sign[x]:
                if pred_return_sign[x] > 0:
                    correct_prob.append([1, positive_return_prob[x]])
                else:
                    correct_prob.append([-1, negative_return_prob[x]])
            else:
                if pred_return_sign[x] > 0:
                    incorrect_prob.append([1, positive_return_prob[x]])
                else:
                    incorrect_prob.append([-1, negative_return_prob[x]])
        #append results to datamodel as attributes
        setattr(data, "correct_prob", correct_prob)
        setattr(data, "incorrect_prob", incorrect_prob)
    
    @staticmethod
    def compute_pl_backtest(data=None, initial_capital=None, bet_size=None,
                            upper_bound=None, lower_bound=None):
        log_return = list(data.model.log_return)
        actual_returns_sign = list(data.model.log_return_sign)
        pred_return_sign = list(data.model.log_pred)
        positive_return_prob = list(data.pred_prob[:, 1])
        negative_return_prob = list(data.pred_prob[:, 0])
        pl_tracker = []
        pl_tracker.append(initial_capital)
        for x in range(len(actual_returns_sign)):
            if pred_return_sign[x] == actual_returns_sign[x]:
                if pred_return_sign[x] > 0:
                    if lower_bound < positive_return_prob[x] < upper_bound:
                        pl = bet_size * initial_capital * log_return[x] * 100
                    else:
                        pl = 0
                elif pred_return_sign[x] < 0:
                    if lower_bound < negative_return_prob[x] < upper_bound:
                        pl = bet_size * initial_capital * -log_return[x] * 100
                    else:
                        pl = 0
                else:
                    pl = 0
            else:
                if pred_return_sign[x] > 0:
                    if lower_bound < positive_return_prob[x] < upper_bound:
                        pl = bet_size * initial_capital * log_return[x] * 100
                    else:
                        pl = 0
                elif pred_return_sign[x] < 0:
                    if lower_bound < negative_return_prob[x] < upper_bound:
                        pl = bet_size * initial_capital * -log_return[x] * 100
                    else:
                        pl = 0
                else:
                    pl = 0
            pl_tracker.append(pl)
        setattr(data, "pl_tracker", pl_tracker)
        setattr(data, "pl_cumsum", np.cumsum(pl_tracker))
                        

        
        
    