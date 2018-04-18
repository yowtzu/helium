from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
import logging
import cvxpy as cvx

class BasePolicy(object):
    """Base class for a trading policy."""
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def get_trades(self, t, h: pd.Series):
        v = sum(h)
        return v * self.get_weights(t, h, None)
 
    @abstractmethod
    def get_weights(self, t, w: pd.Series, v: float):
        return NotImplementedError

class Hold(BasePolicy):
    """Hold initial holding"""
    pass

class MarketCapWeighted(BasePolicy):
    
    def __init__(self, w_benchmark, time_steps):
        self.w_benchmark = w_benchmark
        self.time_steps = time_steps
        super(ProportionalTrade, self).__init__()

    def get_weights(self, t, w: pd.Series, v: float):
        z = self.w_benchmark.loc[t] - w
        return z

class SinglePeriodOpt(BasePolicy):
    
    def __init__(self, rets, costs, constraints):
        super().__init__()
  
        self.rets = rets
        self.costs = costs
        self.constraints = constraints

    def get_weights(self, t, h: pd.Series, v: float):
        v = sum(h)
        w = h / v
        z = cvx.Variable(w.size)
        w_plus = w.values + z 
        ### Equation 4.4 & 4.5
        ret = self.rets.expr(t, w_plus, z, v, t)
        assert(ret.is_concave())
        costs = [ cost.expr(t, w_plus, z, v, t) for cost in self.costs ] 
        for cost in costs:
            assert(cost.is_convex())
        constraints = [ const.expr(t, w_plus, z, v, t) for const in self.constraints ] 
        constraints += [ cvx.sum_entries(z) == 0. ] 
        for constraint in constraints:
            assert(constraint.is_dcp())

        ### Problem
        print('******')
        print("QQQ:", h)
        obj = ret - sum(costs)
        print("Obj: {}".format(obj))
        print("constraints: {}".format(constraints))
        prob = cvx.Problem(cvx.Maximize(obj), constraints)
        
        z_res = pd.Series(index=w.index, data = 0.0)
        try:
            prob.solve()
            if prob.status == cvx.UNBOUNDED:
                logging.error('The problem is unbounded')
            elif prob.status == cvx.INFEASIBLE:
                logging.error('The problem is infeasible')
            else:
                z_res = pd.Series(index=w.index, data =z.value.A1)
        except cvx.SolverError:
            logging.error('The solver failed')
        print(z_res)
        print('******')
        return z_res

def MultiPeriodOpt():
    def __init__(trading_times):
        self.trading_times = trading_times

    def get_weights(self, t, w: pd.Series, v: float):
        raise NotImplementedError
