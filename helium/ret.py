from abc import ABCMeta, abstractmethod
import cvxpy as cvx
import pandas as pd

__all__ = [ 'DefaultRet', ]

class BaseRet(object):
    """Forecasted returns
    """
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.w_benchmark = kwargs.pop('w_benchmark', 0.)
        self.cash_ticker = kwargs.pop('cash_ticker', '_CASH')

    def expr(self, t, w_plus, z, v, theta=0):
        """Returns the estimate at time t of alpha at time tau.

        Args:
            t: time
            w_plus: post-trade weights
            z: trade weights
            v: portfolio dollar value
            tau: prediction target time. if tau=None means t
            theta: int: how many extra step extra to predict, default to 0 for single period
        """
        #TO DO to put back benchmark
        return self._expr(t, w_plus, z, v, theta)

    @abstractmethod
    def _expr(self, t, w_plus, z, v, theta):
         raise NotImplementedError

class DefaultRet(BaseRet):
    def __init__(self, rets, deltas, gamma_decay, **kwargs):
        """
        Args:
            rets: dictionary as_of_time: dataframe(forecast_time, ticker)
            deltas: dictionary as_of_time: dataframe(forecast_time, ticker)
        """
        super(DefaultRet, self).__init__(**kwargs)
        self.rets = rets
        self.deltas = deltas
        self.gamma_decay = gamma_decay

    def _expr(self, t, w_plus, z, v, theta):
        rets = self.rets[t].iloc[theta].values
        deltas = self.deltas[t].iloc[theta].values
        #rets = self.rets.loc[t].values
        #deltas = self.deltas.loc[t].values
        alpha = cvx.mul_elemwise(rets, w_plus)
        alpha -= cvx.mul_elemwise(deltas, cvx.abs(w_plus))
        estimate = cvx.sum_entries(alpha)
        #estimate = w_plus.T * rets - cvx.abs(w_plus.T) * deltas
        if theta > 0  and self.gamma_decay is not None:
            estimate *= theta**(-self.gamma_decay)
        return estimate
