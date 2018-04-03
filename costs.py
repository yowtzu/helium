class HoldingCost():
    """Model the holding costs"""
    def __init__(borrow_costs, dividends, cash_ticker='CASH'):
        self.borrow_costs = borrow_costs
        self.dividends = dividends
        self.cash_ticker = cash_ticker

    def _estimate(self, t, w_plus, z, value):
        w_plus = w_plus[w_plus.index != self.cash_key]
        w_plus = w_plus.values
        return cvx.sum_entries( cvx.mul_elemwise() + cvx.mul_elemenwise() )

    def value_expr(self, t, h_plus, u):
        pass

class TransactionCost():
    def __init__(half_spread, nonlin_coef, sigma, nonlin_power, volume, asym_coef):
        self.half_spread = half_spread
        self.nonlin_coef = nonlin_coef
        self.sigma = sigma
        self.nonlin_power = nonlin_power
        self.volume = volume
        self.asym_coef = asym_coef

    def _estimate():
        """Estimate transaction cost"""
        pass

     def value_expr(self, t, h_plus, u):
        pass
 
