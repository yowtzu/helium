from abc import ABCMeta, abstractmethod
import cvxpy as cvx
import pandas as pd

__all__ = [ 'TradeLimitConstraint', 'LongOnlyConstraint', 'LeverageLimitConstraint', 'MinCashBalanceConstraint', 'MaxTradeConstraint', ]

class BaseConstraint(object):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.w_benchmark = kwargs.pop('w_benchmark', 0.)
        self.cash_ticker = kwargs.pop('cash_ticker', '_CASH')

    def expr(self, t, w_plus, z, v, theta):
        """Create a list of optimisation constraints

        Args:
            t: time
            w_plus: post-trade weights
            z: trade weights
            v: portfolio dollar value
        """
        return self._expr(t, w_plus - self.w_benchmark, z, v, theta)

    @abstractmethod
    def _expr(self, t, w_plus, z, v):
         raise NotImplementedError

class TradeLimitConstraint(BaseConstraint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _expr(self, t, w_plus, z, v, theta):
        z = z.copy()
        z[self.cash_ticker] = 0.
        return v * cvx.abs(z) <= self.ADVs.loc[t].values * self.max_fraction

class LongOnlyConstraint(BaseConstraint):
    """Constraint on Long only, i.e., weights >=0"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _expr(self, t, w_plus, z, v, theta):
        return w_plus >= 0

class LeverageLimitConstraint(BaseConstraint):
    """Constraint on leverage"""

    def __init__(self, limit, **kwargs):
        self.limit = limit
        super().__init__(**kwargs)

    def _expr(self, t, w_plus, z, v, theta):
        return cvx.norm(w_plus, 1) <= self.limit

class MinCashBalanceConstraint(BaseConstraint):
    def __init__(self, threshold, **kwargs):
        self.threshold = threshold 
        super().__init__(**kwargs)

    def _expr(self, t, w_plus, z, v, theta):
        return w_plus['_CASH'] >= self.threshold

class MaxTradeConstraint(BaseConstraint):
    """Constraint on maximum trainding size"""

    def __init__(self, dollar_volume, max_fraction=0.05, **kwargs):
        """
        Args:
            dollar_volume: pd.DataFrame with tickers as columns and time as index
        """
        self.dollar_volume = dollar_volume
        self.max_fraction = max_fraction
        super().__init__(**kwargs)

    def _expr(self, t, w_post, z, v, theta):
        z = z.copy()
        z[self.cash_ticker] = 0.
        return cvx.abs(z) * v <= self.max_fraction * self.dollar_volume.loc[t].values
