import copy
import logging
import time

import multiprocess
import numpy as np
import pandas as pd
import cvxpy as cvx
from collections import OrderedDict
from typing import NamedTuple
from .ret import BaseRet
from .result import Result
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
        h_plus = h + u

        costs =[ cost.value_expr(t, h_plus / v, z, v, t) for cost in self.costs ]
        for cost in costs:
            assert(not pd.isnull(cost))
            assert(not np.isinf(cost))
        
        u[self.cash_ticker] = -(sum(u[u.index != self.cash_ticker]) - sum(costs))
        h_plus[self.cash_ticker] = h[self.cash_ticker] + u[self.cash_ticker]

        h_next = (1 + self.rets.loc[t]) * h_plus
        return h_next, u
         
    def run(self, h_init: pd.Series, policy, start_date, end_date, **kwargs):
        """Backtest a single policy"""
        h = h_init.copy()
        results = Result(initial_portfolio=h_init.copy(),
            policy=policy, cash_key=self.cash_ticker,
            simulator=self)
        

        dates = self.rets[start_date:end_date].index
        logging.info('Backtest started, from {start_date} to {end_date}'.format(start_date=dates[0], end_date=dates[-1]))

        h_ts = [h.copy()]
        for t, ret in self.rets.iterrows():
            u = pd.Series()
            logging.info('Getting trades at date: {date}'.format(date=t))
            start = time.time()
            try:
                u = policy.get_trades(t, h)
            except cvx.SolverError:
                logging.warning('Solver failed on time %s. Default to no trades.' % t)

            logging.info('Propagating portfolio at time %s' % t)
            start = time.time()
            h_plus, u  = self.step(t, h, u)
            end = time.time()
            h = h_plus
            h_ts.append(h)
            results.log_simulation(t=t, u=u, h_next=h,
                                   risk_free_return=self.rets.loc[t, self.cash_ticker],
                                   exec_time=end - start)
            
        h_ts = pd.concat(h_ts, axis=1)
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

