from abc import ABCMeta, abstractmethod
import cvxpy as cvx

__all__ = [ 'HoldingCost', 'TransactionCost', ]

class BaseCost(object):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.cash_ticker = kwargs.pop('cash_ticker', '_CASH')

    def estimate_unnormalised(self, t, h, u):
        """A cvx expression that represents the cost in dollar value
            t: current time
            h_plus: unnoramlised pre-trade weights
            u: unnormalised pre-trade trade weights
            v: pre-trade portfolio dollar value
            tau: prediction time
        """
        v = sum(h)
        z = u / v
        w_plus = (h + u) / v
        return v * self.estimate(t, w_plus, z, v, tau)
    
    def estimate(self, t, w_plus, z, v, tau):
        """ A cvx expression that represents the cost

        Args:
            t: current time
            w_plus: post-trade weights
            z: pre-trade trade weights
            v: pre-trade portfolio dollar value
            tau: prediction time
        """
        
        return self._estimate(self, t, w_plus, z, v, tau)

    @abstractmethod
    def _estimate(self, t, w_plus, z, v, tau):
         raise NotImplementedError

class HoldingCost(BaseCost):
    """Model the holding costs"""
    def __init__(self, gamma, borrow_costs, dividends, **kwargs):
        """
            Args:
                gamma = float
                borrow_costs = pd.DataFrame
                dividends = pd.DataFrame
        """
        super(HoldingCost, self).__init__(**kwargs)
        self.gamma = gamma
        self.borrow_costs = borrow_costs
        self.dividends = dividends

    def _estimate(self, t, w_plus, z, v, tau):
        """Estimate holding cost"""
        # w_plus = w_plus.copy()
        # w_plus[self.cash_ticker] = 0.
        borrow_costs = self.borrow_costs.loc[t].values
        dividends = self.dividends.loc[t].values
        cost = cvx.neg(w_plus) * borrow_costs - w_plus * dividends
        return gamma * cvx.sum_entries(cost)

class TransactionCost(BaseCost):
    def __init__(self, gamma, half_spread, nonlin_coef, sigmas, nonlin_power, volumes, asym_coef, **kwargs):
        """
            Args:
                gamma = float
                half_spread = float
                nonlin_coef = float
                sigmas = price volatility in pd.DataFrame
                nonlin_power = float
                volumes = ADV in pd.DataFrame
                asym_coef = float
        """
        super(TransactionCost, self).__init__(**kwargs)
        self.gamma = gamma
        self.half_spread = half_spread
        self.nonlin_coef = nonlin_coef
        self.sigmas = sigmas
        self.nonlin_power = nonlin_power
        self.volumes = volumes
        self.asym_coef = asym_coef

    def _estimate(self, t, w_plus, z, v, tau):
        """Estimate transaction cost"""
        z = z.copy()
        z[self.cash_ticker] = 0.
        z_abs = cvx.abs(z)
        sigma = self.sigmas.loc[t].values
        volumes = self.volumes.loc[t].values
        p = self.power
        cost =  half_spread * z_abs + \
            self.nonline_coef * sigmas * z_abs**self.power * (v / volumes)**(p-1) + \
            self.asym_coef * z
        return gamma * cvx.sum_entries(cost)
