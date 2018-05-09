import logging
import time

import multiprocess
import numpy as np
import pandas as pd
import cvxpy as cvx
from .results import Result

__all__ = ["MarketSimulator", ]


class MarketSimulator:
    """Simulate the financial market for a given strategy"""

    def __init__(self, returns, volumes, costs, cash_ticker='_CASH', **kwargs):
        log_level = kwargs.pop('log_level', logging.WARN)
        logging.basicConfig(level=log_level)

        """Args:
            cash_ticker: the cash_ticker
        """
        self.returns = returns
        self.volumes = volumes
        self.costs = costs
        self.cash_ticker = cash_ticker

    def step(self, t, h, u):
        """Run the portfolio at one time period t, given the current holding h and trades u

        Args:
            h: pandas Series object describing current holding
            u: pandas Series object describing trades
            t: current time

        Returns:
            h_next: portfolio after returns propagation
            u: trades vector with simulated cash balance
        """
        assert (h.index.equals(u.index))

        # don't trade if volume is null
        null_trades = self.volumes.columns[self.volumes.loc[t] == 0]
        if len(null_trades):
            logging.info('No trade condition for stocks %s on %s'.format(null_trades, t))
            u.loc[null_trades] = 0.

        h_plus = h + u
        costs = [cost.realised_value(t, h_plus, u) for cost in self.costs]

        logging.info("t={}".format(t))
        logging.info("h={}".format(h))
        logging.info("u={}".format(u))
        logging.info("costs={}".format(costs))

        for cost in costs:
            assert (not pd.isnull(cost))
            assert (not np.isinf(cost))

        u[self.cash_ticker] = -sum(u[u.index != self.cash_ticker]) - sum(costs)
        h_plus[self.cash_ticker] = h[self.cash_ticker] + u[self.cash_ticker]

        h_next = self.returns.loc[t] * h_plus + h_plus

        return h_next, u

    def run(self, h_init: pd.Series, policy, start_date, end_date):
        """Run a back-testing for a given policy"""
        h = h_init.copy()
        results = Result(initial_portfolio=h_init.copy(), policy=policy, cash_key=self.cash_ticker, simulator=self)

        self.returns = self.returns[start_date:end_date]
        dates = self.returns.index
        logging.info(
            'Back-test started, from {start_date} to {end_date}'.format(start_date=dates[0], end_date=dates[-1]))

        for t, ret in self.returns.iterrows():
            u = pd.Series()
            logging.info('Getting trades at date: {date}'.format(date=t))
            try:
                u = policy.get_trades(t, h)
            except cvx.SolverError:
                logging.warning('Solver failed on time %s. Default to no trades.'.format(t))

            logging.info('Propagating portfolio at time %s'.format(t))
            start = time.time()
            h_plus, u = self.step(t, h, u)
            end = time.time()
            logging.info("t={}".format(t))
            logging.info("h={}".format(h))
            h = h_plus

            results.log_simulation(t=t, u=u, h_next=h,
                                   risk_free_return=self.returns.loc[t, self.cash_ticker],
                                   exec_time=end - start)

        return results

    def run_multi(self, h_init, policies, start_date, end_date, parallel=True, **kwargs):

        def _run(policy):
            return self.run(h_init, policy, start_date, end_date)

        if parallel:
            cpu_count = kwargs.pop('cpu_count', multiprocess.cpu_count())
            process_pool = multiprocess.Pool(processes=cpu_count)
            results = process_pool.map(_run, policies)
            return results
        else:
            return list(map(_run, policies))
