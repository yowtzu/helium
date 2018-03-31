from abc import ABCMeta, abstractmethod
import cvxpy as cvx
import pandas as pd

__all__ == ['LongOnly', 'LeverageLimit', 'LongCash', 'MaxTrade']

class BaseConstraint(object):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.w_benchmark = kwargs.pop('w_benchmark', 0.)

    def weight_expr(self, t, w_post, z, v):
        """Create a list of optimisation constraints

        Args:
            t: time
            w_post: post-trade weights
            z: trade weights
            v: portfolio dollar value
        """
        return self._weight_expression(self, t, w_post - self.w_benchmark, z, v)


    @abstractmethod
    def _weight_expr(self, t, w_post, z, v):
        """Create a list of optimisation constraints

        Args:
            t: time
            w_post: post-trade weights
            z: trade weights
            v: portfolio dollar value
        """
        pass

class LongOnly(BaseConstraint):
    """Constraint on Long only, i.e., weights >=0"""

    def __init__(self, **kwargs):
        super(LongOnly, self).__init__(**kwargs)

    def _weight_expr(self, t, w_post, z, v):
        return w_post >= 0

class LeverageLimit(BaseConstraint):
    """Constraint on leverage"""

    def __init__(self, limit, **kwargs):
        self.limit = limit
        super(LeverageLimit, self).__init__(**kwargs)

    def _weight_expr(self, t, w_post, z, v):
        return cvx.norm(w_post, 1) <= limit

class LongCash(BaseConstraint):
    def __init__(self, **kwargs):
        super(LongCash, self).__init__(**kwargs)

    def _weight_expr(self, t, w_post, z, v):
        return w_post[-1] >= 0.

class MaxTrade(BaseConstraint):
    """Constraint on maximum trainding size"""

    def __init__(self, ADVs, max_fraction=0.05, **kwargs):
        self.ADVs = ADVs
        self.max_fraction = max_fraction
        super(MaxTrade, self).__init__(**kwargs)

    def _weight_expr(self, t, w_post, z, v):
        cvx.abs(z) * v <= ADVs * self.max_fraction
