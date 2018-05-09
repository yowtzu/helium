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

    def expr(self, t, w_plus, z, v, theta=0):
        """Returns the estimate at time t of alpha at time tau.

        Args:
            t: time
            w_plus: post-trade weights
            z: trade weights
            v: portfolio dollar value
            theta: int: how many extra step extra to predict, default to 0 for single period
        """
        # TO DO to put back benchmark
        return self._expr(t, w_plus, z, v, theta)

    @abstractmethod
    def _expr(self, t, w_plus, z, v, theta):
        return


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

    def _expr(self, t, w_plus, z, v, theta):
        returns = self.returns[t].iloc[theta].values
        deltas = self.deltas[t].iloc[theta].values

        #####
        # alpha = cvx.mul_elemwise(rets, w_plus)
        # alpha -= cvx.mul_elemwise(deltas, cvx.abs(w_plus))
        # estimate = cvx.sum_entries(alpha)
        ######
        # estimate = rets.T * w_plus
        #####

        estimate = returns.T * w_plus - cvx.abs(w_plus.T) * deltas

        if theta > 0 and self.gamma_decay is not None:
            estimate *= theta ** (-self.gamma_decay)
        return estimate
