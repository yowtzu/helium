from abc import ABC, abstractmethod
import cvxpy as cvx

__all__ = ['TradeLimitConstraint', 'LongOnlyConstraint', 'LeverageLimitConstraint', 'MinCashBalanceConstraint', 'TurnoverConstraint', 'MaximumHolding']


class BaseConstraint(ABC):

    def __init__(self, w_benchmark=0., cash_ticker='_CASH', **kwargs):
        """Args:

            w_benchmark:  benchmark weight single column data frame
            cash_ticker: the cash_ticker
        """
        self.w_benchmark = w_benchmark
        self.cash_ticker = cash_ticker

    def expr(self, t, w_plus, z, v, theta):
        """Create optimisation cvx constraint expression

        Args:
            t: time
            w_plus: post-trade weights
            z: trade weights
            v: portfolio dollar value
            theta: int: how many extra step extra to predict, default to 0 for single period
        """
        return self._expr(t, w_plus - self.w_benchmark, z, v, theta)

    @abstractmethod
    def _expr(self, t, w_plus, z, v, theta):
        return


class TradeLimitConstraint(BaseConstraint):

    def __init__(self, avg_daily_volume, max_fraction=0.05, **kwargs):
        """Args:

            avg_daily_volume: pd.DataFrame with tickers as columns and time as index
            max_fraction:  abs(z) * volume <= max_fraction * dollar volume
        """
        self.avg_daily_volume = avg_daily_volume
        self.max_fraction = max_fraction
        super().__init__(**kwargs)

    def _expr(self, t, w_plus, z, v, theta):
        z = z.copy()
        z[self.cash_ticker] = 0.
        return v * cvx.abs(z) <= self.max_fraction * self.avg_daily_volume.loc[t].values


class LongOnlyConstraint(BaseConstraint):
    """Constraint on Long only, i.e., each holding weight >=0"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _expr(self, t, w_plus, z, v, theta):
        return w_plus >= 0


class LeverageLimitConstraint(BaseConstraint):
    """Constraint on leverage"""

    def __init__(self, limit, **kwargs):
        """
        Args:
            limit: multiple of portfolio size
        """
        self.limit = limit
        super().__init__(**kwargs)

    def _expr(self, t, w_plus, z, v, theta):
        return cvx.norm(w_plus, 1) <= self.limit

class MinCashBalanceConstraint(BaseConstraint):
    """Constraint on minimum amount of cash on holding at each step"""

    def __init__(self, threshold, **kwargs):
        """Create a list of optimisation constraints

        Args:
            t: time
            w_plus: post-trade weights
            z: trade weights
            v: portfolio dollar value
            theta: prediction step ahead
            threshold: the minimum of cash to hold in percentage term
        """
        self.threshold = threshold
        super().__init__(**kwargs)

    def _expr(self, t, w_plus, z, v, theta):
        return w_plus[self.cash_ticker] >= self.threshold

class TurnoverConstraint(BaseConstraint):
    """Constraint on turnover"""

    def __init__(self, limit, **kwargs):
        """
        Args:
            limit: 2 way turnover in term of percentage of portfolio size
        """
        self.limit = limit
        super().__init__(**kwargs)

    def _expr(self, t, w_plus, z, v, theta):
        return cvx.norm(z, 1) <= self.limit


class MaximumHolding(BaseConstraint):
    """Constraint on minimum amount of cash on holding at each step"""

    def __init__(self, threshold, **kwargs):
        """Create a list of optimisation constraints

        Args:
            t: time
            w_plus: post-trade weights
            z: trade weights
            v: portfolio dollar value
            theta: prediction step ahead
            threshold: the minimum of cash to hold in percentage term
        """
        self.threshold = threshold
        super().__init__(**kwargs)

    def _expr(self, t, w_plus, z, v, theta):
        return w_plus <= self.threshold
