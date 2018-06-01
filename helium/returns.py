from abc import ABC, abstractmethod
import cvxpy as cvx

__all__ = ['DefaultReturns', ]


class BaseReturns(ABC):
    """Estimated returns """

    def __init__(self,  w_benchmark=0., cash_ticker='_CASH', **kwargs):
        """Args:

            w_benchmark:  benchmark weight single column data frame
            cash_ticker: the cash_ticker
        """
        self.w_benchmark = w_benchmark
        self.cash_ticker = cash_ticker

    @abstractmethod
    def expr(self, t, w_plus, z, v, theta):
        """Returns the estimate at time t of alpha at time tau.

        Args:
            t: time
            w_plus: post-trade weights
            z: trade weights
            v: portfolio dollar value
            theta: int: how many extra step extra to predict, default to 0 for single period
        """
        # TO DO to put back benchmark
        pass


class DefaultReturns(BaseReturns):
    def __init__(self, returns, deltas, gamma_decay, **kwargs):
        """
        Args:
            returns: dictionary as_of_time: pd.DataFrame(forecast_time, ticker)
            deltas: dictionary as_of_time: pd.DataFrame(forecast_time, ticker)
        """
        super().__init__(**kwargs)
        self.returns = returns
        self.deltas = deltas
        self.gamma_decay = gamma_decay

    def expr(self, t, w_plus, z, v, theta=0):
        returns = self.returns[t].iloc[theta].values
        deltas = self.deltas[t].iloc[theta].values

        #####
        # alpha = cvx.mul_elemwise(returns, w_plus)
        # alpha -= cvx.mul_elemwise(deltas, cvx.abs(w_plus))
        # estimate = cvx.sum_entries(alpha)
        ######
        # estimate = returns.T * w_plus
        #####

        estimate = cvx.mul_elemwise(returns, w_plus)
        estimate -= cvx.mul_elemwise(deltas, cvx.abs(w_plus))
        estimate = cvx.sum_entries(estimate)
        if theta > 0 and self.gamma_decay is not None:
            estimate *= theta ** (-self.gamma_decay)
        return estimate
