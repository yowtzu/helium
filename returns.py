from abc import ABCMeta, abstractmethod
import cvxpy as cvx
import pandas as pd

class BaseReturnForecast(object):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.w_benchmark = kwargs.pop('w_benchmark', 0.)
        self.cash_ticker = kwargs.pop('cash_ticker', '_CASH')

    def estimate(self, t, w_plus, z, v, tau):
        """Returns the estimate at time t of alpha at time tau.

        Args:
            t: time
            w_plus: post-trade weights
            z: trade weights
            v: portfolio dollar value
            tau: prediction target time. if tau=None means t
        """`
        if tau is None:
            tau = t
        return self._estimate(self, t, w_plus - self.w_benchmark, z, v, tau)

    @abstractmethod
    def _expr(self, t, w_plus, z, v):
         raise NotImplementedError

class DefaultReturnForecast(BaseReturnForecast):
    def __init__(self, returns, gamma_decay, **kwargs):
        super(SPReturnForecast, self).__init__(**kwargs)
        self.returns = returns
        self.gamma_decay = gamma_decay

    def _estimate(self, t, w_plus, z, v, tau):
        returns = returns.loc[t].values
        estimate = cvx.sum_entries(w_plus * self.returns - w_plus * cvx.abs(self.delta))
        if tau > t and self.gamma_decay is not None:
            estimate * = (tau - t)**(-self.gamma_decay)

class ReturnsForecast(BaseReturnForecast):
    """A single alpha estimation.

    Attributes:
      return_estimates: A Multi-Index dataframes of return estimates.
    """

    def __init__(self, return_estimates, **kwargs):
        super(MPReturnsForecast, self).__init__(**kwargs)
        self.return_estimates = return_estimates

    def _expr(self, t, w_plus, z, v, tau):
        return self.return_estimates.iloc[(t, tau)].values.T * w_plus
