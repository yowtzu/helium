from abc import ABCMeta, abstractmethod
import cvxpy as cvx
import pandas as pd

__all__ = [ 'DefaultRet', ]

class BaseRet(object):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.w_benchmark = kwargs.pop('w_benchmark', 0.)
        self.cash_ticker = kwargs.pop('cash_ticker', '_CASH')

    def expr(self, t, w_plus, z, v, tau):
        """Returns the estimate at time t of alpha at time tau.

        Args:
            t: time
            w_plus: post-trade weights
            z: trade weights
            v: portfolio dollar value
            tau: prediction target time. if tau=None means t
        """
        if tau is None:
            tau = t
        return self._expr(t, w_plus - self.w_benchmark, z, v, tau)

    @abstractmethod
    def _expr(self, t, w_plus, z, v, tau):
         raise NotImplementedError

class DefaultRet(BaseRet):
    def __init__(self, rets, deltas, gamma_decay, **kwargs):
        super(DefaultRet, self).__init__(**kwargs)
        self.rets = rets
        self.deltas = deltas
        self.gamma_decay = gamma_decay

    def _expr(self, t, w_plus, z, v, tau):
        rets = self.rets.loc[t].values
        deltas = self.deltas.loc[t].values
        estimate = w_plus.T * rets - w_plus.T * cvx.abs(deltas)
        if tau > t and self.gamma_decay is not None:
            estimate *= (tau - t)**(-self.gamma_decay)
        return estimate
    
class RetForecast(BaseRet):
    """A single alpha estimation.

    Attributes:
      return_estimates: A Multi-Index dataframes of return estimates.
    """

    def __init__(self, return_estimates, **kwargs):
        super(MPReturnsForecast, self).__init__(**kwargs)
        self.return_estimates = return_estimates

    def _expr(self, t, w_plus, z, v, tau):
        return self.return_estimates.iloc[(t, tau)].values.T * w_plus
