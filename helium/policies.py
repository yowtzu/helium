from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
import logging
import cvxpy as cv

class BasePolicy(object):
    """Base class for a trading policy."""
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def get_trades(self, holding, t):
        return NotImplemented

class Hold(BasePolicy):
    """Hold initial holding"""
    
    def get_trades(self, holding, t):
        return pd.Series()

class MarketCapWeighted(BasePolicy):
    
    def __init__(self, w_benchmark, time_steps):
        self.w_benchmark = w_benchmark
        self.time_steps = time_steps
        super(ProportionalTrade, self).__init__()

    def get_trades(self, holding, t):
        trades = self.w_benchmark.loc[t] - holding / sum(holding)
        return sum(holding) * trades

class SinglePeriodOpt(BasePolicy):
    
    def __init__(self, return_forecast, costs, constraints):
        super(SinglePeriodOpt, self).__init__()
  
        self.return_forecast = return_forecast
        self.costs = costs
        self constraints = constraints


    def get_trades(self, holding):
        w = holding  / holding.sum()
        z = cvx.Variable(w.size)
        w_plus = w.values + z 
        
        alpha_term = self.return_forecast.weight_expr()

        assert(alpha_term.is_concave())
        assert()

        obj = 
        constraints = 

        prob = cvx.Problem(obj, constraints)
          
        trades = pd.Series()
        try:
            prob.solve()
            if prob.status == cvx.UNBOUNDED:
                logging.error('The problem is unbounded')
            elif prob.status == cvx.INFEASIBLE:
                logging.error('The problem is infeasible')
            else:
                trades = pd.Series(index=holding.index, data = z.value.A1 * value)
        except cvx.SolverError:
            logging.error('The solver failed')
        
        return trades

def MultiPeriodOpt():
    def __init__(trading_times):
        self.trading_times = trading_times
        pass

    def get_trades(self, holding):
        w = holding/holding.sum()
        
        prob_list = []
        z_list = []

        for tau in self.trading_times[t:t+5]:
            fljffjklfj
