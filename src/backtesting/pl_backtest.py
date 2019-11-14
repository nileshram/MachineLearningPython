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
            if initial_capital >= 0:
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
            else:
                pl = 0
            initial_capital += pl
            pl_tracker.append(pl)
            setattr(data, "pl_tracker", pl_tracker)
            setattr(data, "pl_cumsum", np.cumsum(pl_tracker))

class PLBacktestingEngine:
    
    def __init__(self):
        pass
    
    def _init_params(self, data):
        self.log_return = list(data.model.log_return)
        self.actual_returns_sign = list(data.model.log_return_sign)
        self.pred_return_sign = list(data.model.log_pred)
        self.positive_return_prob = list(data.pred_prob[:, 1])
        self.negative_return_prob = list(data.pred_prob[:, 0])
    
    def run_backtest(self, data=None, initial_capital=None, bet_size=None, upper_bound=None, lower_bound=None):
        self._init_params(data)
        pl_tracker = self._run(initial_capital=initial_capital, bet_size=bet_size, upper_bound=upper_bound,
                               lower_bound=lower_bound)
        setattr(data, "pl_tracker", pl_tracker)
        setattr(data, "pl_cumsum", np.cumsum(pl_tracker))

    def _run(self, initial_capital=None, bet_size=None,
                            upper_bound=None, lower_bound=None):
        pl_tracker = []
        pl_tracker.append(initial_capital)
        for x in range(len(self.actual_returns_sign)):
            if initial_capital >= 0:
                if self.pred_return_sign[x] > 0:
                    if lower_bound < self.positive_return_prob[x] < upper_bound:
                        pl = bet_size * initial_capital * self.log_return[x] * 100
                    else:
                        pl = 0
                elif self.pred_return_sign[x] < 0:
                    if lower_bound < self.negative_return_prob[x] < upper_bound:
                        pl = bet_size * initial_capital * -self.log_return[x] * 100
                    else:
                        pl = 0
            else:
                pl = 0
            initial_capital += pl
            pl_tracker.append(pl)
        return pl_tracker
        
        
    