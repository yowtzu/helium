import copy
import logging
import time

import multiprocess
import numpy as np
import pandas as pd
import cvxpy as cvx
from collections import OrderedDict
from .ret import BaseRet

__all__ = [ "MarketSimulator" ]

class MarketSimulator():
    """Simulate the financial market for a given strategy"""

    def __init__(self, rets, volumes, costs, **kwargs):
        log_level = kwargs.pop('log_level', logging.INFO)
        logging.basicConfig(level=log_level)
        
        self.rets = rets
        self.volumes = volumes
        self.costs = costs
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
        
        u[self.cash_ticker] = -(sum(u[u.index != self.cash_ticker]) - sum(cost))
        h_plus[self.cash_ticker] = h[self.cash_ticker] + u[self.cash_ticker]

        h_next = (1 + self.rets.loc[t]) * h_plus
        return h_next, u
    
    
        h_plus = self.returns.loc[t].values * h_plus  + h_plus
        return h_plus, u
         
    def run(self, h_init, policy, start_date, end_date, **kwargs):
        """Backtest a single policy"""

        result = OrderedDict()
        h = h_init
        
        dates = self.rets[start_date:end_date].index
        logging.info('Backtest started, from %s to %s' % (dates[0], dates[-1]))

         
        for t, ret in self.rets.iterrows():
            u = pd.Series()
            logging.info('Getting trades at time %s' % time)
            start = time.time()
            try:
                u = policy.get_trades(h, t)
            except cvx.SolverError:
                logging.warning('Solver failed on time %s. Default to no trades.' % t)

            logging.info('Propagating portfolio at time %s' % t)
            h_plus, u  = self.step(t, h, u)
            results[t] = (h.copy(), u, copy(), h_plus.copy())
            h = h_plus

        return results

    def run_multi(self, h_init, policies, start_time, end_time, parallel, **kwargs):
        def _run(policy):
            return self.run(h_init, policy, start_time, end_time)

        if parallel:
            processes = kwargs.pop('processor', multiprocess.cpu_count())
            process_pool = multiprocess.Pool(processes)
            results = process_pool.map(_run, policies)
            return results
        else:
            return list(map(_run, policies))

