from abc import ABCMeta, abstractmethod
import cvxpy as cvx
import numpy as np

__all__ = [ 'BasicRiskCost', 'FactorRiskCost', 'HoldingCost', 'TransactionCost', ]

class BaseCost(object):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.cash_ticker = kwargs.pop('cash_ticker', '_CASH')

    #def estimate_unnormalised(self, t, h, u):
    #    """A cvx expression that represents the cost in dollar value
    #        t: current time
    #        h_plus: unnoramlised pre-trade weights
    #        u: unnormalised pre-trade trade weights
    #        v: pre-trade portfolio dollar value
    #        tau: prediction time
    #    """
    #    v = sum(h)
    #    z = u / v
    #    w_plus = (h + u) / v
    #    return v * self.estimate(t, w_plus, z, v, t)
    
    def expr(self, t, w_plus, z, v, theta):
        """ A cvx expression that represents the cost

        Args:
            t: current time
            w_plus: post-trade weights
            z: pre-trade trade weights
            v: pre-trade portfolio dollar value
            tau: prediction time
        """
        return self._expr(t, w_plus, z, v, theta)

    @abstractmethod
    def _expr(self, t, w_plus, z, v, theta):
         raise NotImplementedError

#     def value_expr(self, t, w_plus, z, v, tau):
#         return self._value_expr(t, w_plus, z, v, tau)

    @abstractmethod
    def value_expr2(self, t, h_plus, u):
        raise NotImplementedError
    
    @abstractmethod
    def _value_expr(self, t, w_plus, z, v, theta):
        raise NotImplementedError

    def simulation_log(self, t):
        return self.last_cost

class BasicRiskCost(BaseCost):
    def __init__(self, gamma, sigmas, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.sigmas = sigmas
        self.gamma_half_life = kwargs.pop('gamma_half_life', None)
        
    def _expr(self, t, w_plus, z, v, theta):
        gamma_multiplier = 1.
        
        if self.gamma_half_life:
            decay_factor = 2 ** (-1 / self.gamma_half_life)
            # TODO not dependent on days
            gamma_init = decay_factor ** theta #((tau - t).days)
            gamma_multiplier = gamma_init * \
                (1 - decay_factor) / (1 - decay_factor)

        sigma = self.sigmas.loc[t].values
        return gamma_multiplier * self.gamma * cvx.quad_form(w_plus, sigma) 

class FactorRiskCost(BaseCost):
    '''PCA Based Factor risk model'''    
    def __init__(self, gamma, returns, window_size, n_factors, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self._construct_factor_model(returns)
        self.window_size = window_size
        self.n_factors = n_factors

    def _construct_factor_model(self, returns, window_size, n_factors, **kwargs):
        self.factor = {}
        self.sigma = {}
        self.idiosync = {}
        
        def _pca_factor(df):
            t = df.index[-1]
            k = self.n_factors
            second_moments = df.values.T@df.values / df.values.shape[0]
            eigen_values, eigen_vectors = np.linalg.eigh(second_moments)
         
            # eigen_values are returned in ascending oarder
            self.factor[t] = eigen_values[-k:]
            self.sigma[t] = eigen_vectors[:, -k:]
            # residuals
            self.idiosync[t] = eigen_vectors[:,:-k]@np.diag(eigen_values[:-k])@eigen_vectors[:,:-k].T
        
        for d in range(self.window_size, len(returns)):
            _pca_factor(returns.loc[d-window_size:d])
                                                      
    def _expr(self, t, w_plus, z, v, theta):
        factor_risk = self.factor_risk[t]
        factor_loading = self.factor_loading[t]
        idiosync = self.idiosync.loc[t].values
        return self.gamma * cvx.quad_form(w_plus.T * factor_loading, * factor_risk)

class HoldingCost(BaseCost):
    """Model the holding costs"""
    def __init__(self, gamma, borrow_costs, dividends, **kwargs):
        """
            Args:
                gamma = float
                borrow_costs = pd.DataFrame
                dividends = pd.DataFrame
        """
        super().__init__(**kwargs)
        self.gamma = gamma
        self.borrow_costs = borrow_costs
        self.dividends = dividends

    def _expr(self, t, w_plus, z, v, theta):
        """Estimate holding cost"""
        
        #borrow_costs = self.borrow_costs.loc[t].values
        #dividends = self.dividends.loc[t].values
        
        #cost = cvx.mul_elemwise(borrow_costs, cvx.neg(w_plus)) - cvx.mul_elemwise(dividends, w_plus)
        #ones = np.ones_like(dividends)
        #ones[-1] = 0.
        #cost  = cvx.mul_elemwise(ones, cost)
        #return self.gamma * cvx.sum_entries(cost)
    
        w_plus = w_plus[:-1]
        borrow_costs = self.borrow_costs.loc[t].values[:-1]
        dividends = self.dividends.loc[t].values[:-1]
        cost = cvx.mul_elemwise(borrow_costs, cvx.neg(w_plus)) - cvx.mul_elemwise(dividends, w_plus)
        return self.gamma * cvx.sum_entries(cost)
    
#     def _value_expr(self, t, w_plus, z, v, tau):
#         """Estimate holding cost"""
#         borrow_costs = self.borrow_costs.loc[t].values
#         dividends = self.dividends.loc[t].values
#         cost = np.negative(w_plus).T * borrow_costs - w_plus.T * dividends
#         return self.gamma * cost
    
    def value_expr2(self, t, h_plus, u):
        h_plus = h_plus[:-1]
        borrow_costs = self.borrow_costs.loc[t].values[:-1]
        dividends = self.dividends.loc[t].values[:-1]
        self.last_cost = self.gamma * sum(-np.minimum(0, h_plus) * borrow_costs - h_plus * dividends)
        return self.last_cost

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
        
        super().__init__(**kwargs)
        self.gamma = gamma
        self.half_spread = half_spread
        self.nonlin_coef = nonlin_coef
        self.sigmas = sigmas
        self.nonlin_power = nonlin_power
        self.volumes = volumes
        self.asym_coef = asym_coef

    def _expr(self, t, w_plus, z, v, theta):
        """Estimate transaction cost"""
        #TO DO make cash component zero
        z = z[:-1]
        z_abs = cvx.abs(z)
        sigma = self.sigmas.loc[t].values[:-1]
        volumes = self.volumes.loc[t].values[:-1]
        
        # this is a n-1 vector
        cost = cvx.mul_elemwise(self.half_spread, z_abs)
        
        # v is atom
        # volumes is n-1 vector
        # this stage, it is numpy term, hence auto element wise multiplication
        second_term = (self.nonlin_coef * sigma) * (v / volumes)**(self.nonlin_power-1)      
        
        # now it is cvx element wise multiplication
        cost += cvx.mul_elemwise(second_term, cvx.abs(z)**self.nonlin_power)

        # this is a n-1 vector
        cost += self.asym_coef * z

        return self.gamma * cvx.sum_entries(cost)
   
#     def _value_expr(self, t, w_plus, z, v, tau):
#         z_abs = np.abs(z) 
#         sigma = self.sigmas.loc[t].values
#         volumes = self.volumes.loc[t].values
#         cost =  self.half_spread * z_abs + \
#             self.nonlin_coef * sigma * z_abs**self.nonlin_power * (v / volumes)**(self.nonlin_power-1) + \
#             self.asym_coef * z
#         cost_without_cash = cost[:-1]
#         return self.gamma * sum(cost_without_cash)
    
    def value_expr2(self, t, h_plus, u):
        u = u[:-1]
        u_abs = np.abs(u)
        sigma = self.sigmas.loc[t].values[:-1]
        volumes = self.volumes.loc[t].values[:-1]
        cost =  self.half_spread * u_abs + \
            self.nonlin_coef * sigma * u_abs**self.nonlin_power / (volumes)**(self.nonlin_power-1) + \
            self.asym_coef * u
#         print("volume:{}".format(volumes))
#         print("u:{}".format(u))
#         print("cost:{}".format(cost))
        self.last_cost = self.gamma * sum(cost)
        return self.last_cost
   