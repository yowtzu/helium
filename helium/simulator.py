import copy
import logging
import time

import multiprocess
import numpy as np
import pandas as pd
import cvxpy as cvx

#from .ret import BaseRet
#from .costs import BaseCost

__all__ = [ "MarketSimulator" ]

class MarketSimulator():
    """Simulate the financial market for a given strategy"""

    def __init__(self, rets, costs, gamma, constraints, **kwargs):
        log_level = kwargs.pop('log_level', logging.INFO)
        logging.basicConfig(level=log_level)
        
        self.rets = rets
        self.costs = costs
        self.constraints = constraints
        self.cash_ticker = kwargs.pop('cash_ticker', '_CASH')
        
    
    def step(self, h, u, t):
        """Propagates the portfolio forward over time period t, given trades u.

        Args:
            h: pandas Series object describing current portfolio
            u: n vector with the stock trades (not cash)
            t: current time

        Returns:
            h_next: portfolio after returns propagation
            u: trades vector with simulated cash balance
        """
        h_plus = h + u
        costs = [ simulated_hcost.estimate(t, h_plus, u) for cost in self.costs ] 

        h_plus = self.returns.loc[t].values * h_plus  + h_plus
        return h_plus, u
         
    def run(self, holding_init, policies, start_time, end_time, parallel=True, **kwargs):
        # check parameters before the hard work
        dates = rets[start_dates:end_date].index
        assert({start_date, end_date} in dates)
        
        def _run_single(policy):
            holding = holding_init.copy()
            logging.info('Backtest started, from %s to %s' % (simulation_times[0], simulation_times[-1]))

            # do the job
            holding_df = pd.DataFrame()
            trades_df = pd.DataFrame()

            return holding_df, trades_df
             
        
        # generate the combination parameters in tuple format
        if parallel:
            processes = kwargs.pop('processor', multiprocess.cpu_count())
            process_pool = multiprocess.Pool(processes)
            results = process_pool.map(holding, policies)
            return results
        else:
            return list

        return 