from abc import ABCMeta, abstractmethod
import cvxpy as cvx
import pandas as pd

class BaseReturn(object):
    __metaclass__ = ABCMeta

    def __init__(self, returns, **kwargs):
        self.returns = returns
        self.w_benchmark = kwargs.pop('w_benchmark', 0.)
        self.cash_ticker = kwargs.pop('cash_ticker', '_CASH')

    def estimate(self, t, w_plus, z, v, tau):
        """Create a list of optimisation constraints

        Args:
            t: time
            w_plus: post-trade weights
            z: trade weights
            v: portfolio dollar value
        """`
        return self._estimate(self, t, w_plus - self.w_benchmark, z, v, tau)

    @abstractmethod
    def _expr(self, t, w_plus, z, v):
         raise NotImplementedError

class SPReturn(BaseReturn):
    def __init__(self, **kwargs):
        super(SPReturn, self).__init__(**kwargs)

    def _estimate(self, t, w_plus, z, v, tau):
        returns = returns.loc[t].values
        return np.sum(w_plus * returns)

class LongOnly(BaseConstraint):
    """Constraint on Long only, i.e., weights >=0"""

    def __init__(self, **kwargs):
        super(LongOnly, self).__init__(**kwargs)

    def expr(self, t, w_plus, z, v):
        return w_plus >= 0

class LeverageLimit(BaseConstraint):
    """Constraint on leverage"""

    def __init__(self, limit, **kwargs):
        self.limit = limit
        super(LeverageLimit, self).__init__(**kwargs)

    def _expr(self, t, w_plus, z, v):
        return cvx.norm(w_plus, 1) <= limit

class MinCashBalanceConstraint(BaseConstraint):
    def __init__(self, threshold, **kwargs):
        self.threshold = threshold 
        super(LongCash, self).__init__(**kwargs)

    def _expr(self, t, w_plus, z, v):
        return w_plus['_CASH'] >= self.threshold

class MaxTrade(BaseConstraint):
    """Constraint on maximum trainding size"""

    def __init__(self, dollar_volume, max_fraction=0.05, **kwargs):
v * cvx.abs(z) <= self.ADVs.loc[t].values * self.max_fraction
        """
        Args:
            dollar_volume: pd.DataFrame with tickers as columns and time as index
        """
        self.dollar_volume = dollar_volume
        self.max_fraction = max_fraction
        super(MaxTrade, self).__init__(**kwargs)

    def _expr(self, t, w_post, z, v):
        z = z.copy()
        z[self.cash_ticker] = 0.
        return cvx.abs(z) * v <= self.max_fraction * self.dollar_volume.loc[t].values
