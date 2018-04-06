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

    def __init__(self, rets, soft_constraints, hard_constraints, **kwargs):
        log_level = kwargs.pop('log_level', logging.INFO)
        logging.basicConfig(level=log_level)
        
        self.rets = rets
        self.soft_constraints = soft_constraints
        self.hard_constraints = hard_constraints
        self.cash_ticker = kwargs.pop('cash_ticker', '_CASH')
    
    def step(self, t, h, u):
        """Run the portfolio at one time period t, given the current holding h andn trades u

        Args:
            h: pandas Series object describing current holding
            u: pandas Series object describing trades
            t: current time

        Returns:
            h_next: portfolio after returns propagation
            u: trades vector with simulated cash balance
        """
        assert(h.index.equals(u.index))
        v = sum(h)
        z = u / v
        costs = [ cost.estimate_unnormalised(t, h, u) for cost in self.costs ]
        for cost in costs:
            assert(not pd.isnull(cost))
            assert(not np.isinf(cost))
        
        u[self.cash_ticker] = -(sum(u[u.index != self.cash_ticker] - sum(cost))
        h_plus[self.cash_ticker] = h[self.cash_ticker] + u[self.cash_ticker]

        h_next = (1 + self.rets.loc[t]) * h_plus
        return h_next, u
    
    
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