from abc import ABCMeta, abstractmethod
import cvxpy as cvx
import pandas as pd

<<<<<<< HEAD:helium/ret.py
__all__ = [ 'DefaultRet', ] 
=======
__all__ = [ 'DefaultRet', ]

>>>>>>> 74a744c194cddcc54ee1c7c5a9c138fc70b8a427:helium/ret.py
class BaseRet(object):
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
        """
        if tau is None:
            tau = t
        return self._estimate(self, t, w_plus - self.w_benchmark, z, v, tau)

    @abstractmethod
    def _expr(self, t, w_plus, z, v):
         raise NotImplementedError

class DefaultRet(BaseRet):
<<<<<<< HEAD:helium/ret.py
    def __init__(self, returns, gamma_decay, **kwargs):
        super(DefaultRet, self).__init__(**kwargs)
        self.returns = returns
=======
    def __init__(self, ret, gamma_decay, **kwargs):
        super(SPRet, self).__init__(**kwargs)
        self.ret = ret
>>>>>>> 74a744c194cddcc54ee1c7c5a9c138fc70b8a427:helium/ret.py
        self.gamma_decay = gamma_decay

    def _estimate(self, t, w_plus, z, v, tau):
        ret = ret.loc[t].values
        estimate = cvx.sum_entries(w_plus * self.ret - w_plus * cvx.abs(self.delta))
        if tau > t and self.gamma_decay is not None:
            estimate *= (tau - t)**(-self.gamma_decay)

<<<<<<< HEAD:helium/ret.py
class ReturnsForecast(BaseRet):
=======
class RetForecast(BaseRet):
>>>>>>> 74a744c194cddcc54ee1c7c5a9c138fc70b8a427:helium/ret.py
    """A single alpha estimation.

    Attributes:
      return_estimates: A Multi-Index dataframes of return estimates.
    """

    def __init__(self, return_estimates, **kwargs):
        super(MPReturnsForecast, self).__init__(**kwargs)
        self.return_estimates = return_estimates

    def _expr(self, t, w_plus, z, v, tau):
        return self.return_estimates.iloc[(t, tau)].values.T * w_plus