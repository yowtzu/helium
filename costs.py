from abc import ABCMeta, abstractmethod

class BaseCost(object):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.cash_ticker = kwargs.pop('cash_ticker', '_CASH')

    def estimate(self, t, w_plus, z, v):
        """ A cvx expression that represents the cost

        Args:
            t: time
            w_plus: post-trade weights
            z: trade weights
            v: portfolio dollar value
        """
       
        return self._estimate(self, t, w_plus, z, v)
    
    @abstractmethod
    def _estimate(self, t, w_plus, z, v):
         raise NotImplementedError

class HoldingCost(BaseCost):
    """Model the holding costs"""
    def __init__(borrow_costs, dividends, **kwargs):
        super(HoldingCost, self).__init__(**kwargs)
        self.borrow_costs = borrow_costs
        self.dividends = dividends

    def _estimate(self, t, w_plus, z, v):
        w_plus = w_plus.copy()
        w_plus[self.cash_ticker] = 0.
        borrow_costs = self.borrow_costs.loc[t].values
        dividends = self.dividends.loc[t].values
        return (-np.minimum(0., -w_plus)) * borrow_costss + w_plus * dividends

class TransactionCost(BaseCost):
    def __init__(half_spread, nonlin_coef, sigmas, nonlin_power, volumes, asym_coef, **kwargs):
        super(TransactionCost, self).__init__(**kwargs)
        self.half_spread = half_spread
        self.nonlin_coef = nonlin_coef
        self.sigmas = sigmas
        self.nonlin_power = nonlin_power
        self.volumes = volumes
        self.asym_coef = asym_coef

    def _estimate(self, t, w_plus, z, v):
        """Estimate transaction cost"""
        z = z.copy()
        z[self.cash_ticker] = 0.
        z_abs = cvx.abs(z)
        sigma = self.sigmas.loc[t].values
        volumes = self.volumes.loc[t].values
        p = self.power
        return half_spread * z_abs + nonline_coef * sigmas * z_abs**p * (v / volumes)**(p-1) + self.asym_coef * z
