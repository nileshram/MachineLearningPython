'''
Created on 11 Nov 2019

@author: nilesh
'''
import numpy as np
import logging 

class PLBacktestingEngine:
    
    def __init__(self):
        self._logger = logging.getLogger("cqf_logger")
    
    def _init_params(self, data):
        self.log_return = list(data.model.log_return)
        self.actual_returns_sign = list(data.model.log_return_sign)
        self.pred_return_sign = self._get_prediction_(data)
        self.positive_return_prob = list(data.pred_prob[:, 1])
        self.negative_return_prob = list(data.pred_prob[:, 0])
    
    def _get_prediction_(self, data):
        for header in list(data.model):
            if "_pred" in header:
                prediction = header
        return list(data.model[prediction])
    
    def _init_payload(self):
        payload = {"pl_tracker" : [],
                   "long_trades" : [],
                    "short_trades" : []}
        return payload

    def _run(self, data=None, initial_capital=None, bet_size=None,
                            upper_bound=None, lower_bound=None):
        payload = self._init_payload()
        payload["pl_tracker"].append(initial_capital)
        for x in range(len(self.actual_returns_sign)):
            if initial_capital >= 0:
                if self.pred_return_sign[x] > 0:
                    if lower_bound < self.positive_return_prob[x] < upper_bound:
                        pl = bet_size * initial_capital * self.log_return[x]
                        long_trade = (list(data.model.Date)[x], sum(payload["pl_tracker"]) + pl)
                        payload["long_trades"].append(long_trade)
                    else:
                        pl = 0
                elif self.pred_return_sign[x] < 0:
                    if lower_bound < self.negative_return_prob[x] < upper_bound:
                        pl = bet_size * initial_capital * -self.log_return[x]
                        short_trade = (list(data.model.Date)[x], sum(payload["pl_tracker"]) + pl)
                        payload["short_trades"].append(short_trade)
                    else:
                        pl = 0
            else:
                pl = 0
            initial_capital += pl
            payload["pl_tracker"].append(pl)
        return payload
    
    def _pretty_print_trading_stats(self, data):
        self._logger.info("Asset name: {}".format(data.filename))
        self._logger.info("Max Portfolio Value: {}".format(max(data.pl_cumsum)))
        self._logger.info("Value @ End: {}".format(list(data.pl_cumsum)[-1]))
        self._logger.info("Total time in market: {}".format(self._compute_time_in_market(data)))
        self._logger.info("Longest market win streak: {} with P&L {}".format(self._compute_longest_win_streak(data)[0],
                                                                    self._compute_longest_win_streak(data)[1]))

    def _compute_time_in_market(self, data):
        running_total = 0
        for pl in list(data.pl_cumsum):
            if not pl <= 0:
                running_total += 1
        return running_total
    
    def _compute_longest_win_streak(self, data):
        max_streak = 0
        running_streak = 0
        max_pl_accumulated = 0
        running_pl_accumulated = 0
        for pl in range(len(data.pl_cumsum) - 1):
            if data.pl_cumsum[pl] < data.pl_cumsum[pl + 1]:
                running_streak += 1
                running_pl_accumulated += data.pl_tracker[pl + 1]
            else:
                if running_streak > max_streak:
                    max_streak = running_streak
                    max_pl_accumulated = running_pl_accumulated
                    running_streak = 0 #reset running streak
                    running_pl_accumulated = 0
        return max_streak, max_pl_accumulated
            
    def run_backtest(self, data=None, initial_capital=None, bet_size=None, upper_bound=None, lower_bound=None):
        self._init_params(data)
        self._logger.info("Running pl backtest for: {}".format(data.filename))
        payload = self._run(data=data, initial_capital=initial_capital, bet_size=bet_size, upper_bound=upper_bound,
                               lower_bound=lower_bound)
        self._logger.info("Completed running pl backtest for: {}".format(data.filename))
        setattr(data, "pl_tracker", payload["pl_tracker"])
        setattr(data, "pl_cumsum", np.cumsum(payload["pl_tracker"]))
        setattr(data, "long_trade_marker", payload["long_trades"])
        setattr(data, "short_trade_marker", payload["short_trades"])
        self._pretty_print_trading_stats(data=data)
        
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
        
        
    