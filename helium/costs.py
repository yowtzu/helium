from abc import ABC, abstractmethod
import cvxpy as cvx
import numpy as np

__all__ = ['BasicRiskCost', 'FactorRiskCost', 'HoldingCost', 'TransactionCost', ]


class BaseCost(ABC):

    def __init__(self, w_benchmark=0., cash_ticker='_CASH', **kwargs):
        """Args:

            w_benchmark:  benchmark weight single column data frame
            cash_ticker: the cash_ticker
        """
        self.w_benchmark = w_benchmark
        self.cash_ticker = cash_ticker
        self.last_cost = None

    @abstractmethod
    def realised_value(self, t, h_plus, u):
        pass

    @abstractmethod
    def expr(self, t, w_plus, z, v, theta):
        """ A cvx expression that represents the cost

        Args:
            t: time
            w_plus: post-trade weights
            z: trade weights
            v: portfolio dollar value
            theta: int: how many extra step extra to predict, default to 0 for single period
        """
        pass

    def simulation_log(self, t):
        return self.last_cost


class BasicRiskCost(BaseCost):
    def __init__(self, gamma, sigmas, gamma_half_life=None, **kwargs):
        self.gamma = gamma
        self.sigmas = sigmas
        self.gamma_half_life = gamma_half_life
        super().__init__(**kwargs)

    def realised_value(self, t, h_plus, u):
        # TO DO
        return 0.

    def expr(self, t, w_plus, z, v, theta):
        gamma_multiplier = 1.
        
        if self.gamma_half_life:
            decay_factor = 2 ** (-1 / self.gamma_half_life)
            gamma_init = decay_factor ** theta
            gamma_multiplier = gamma_init * \
                (1 - decay_factor) / (1 - decay_factor)

        sigma = self.sigmas.loc[t].values
        return gamma_multiplier * self.gamma * cvx.quad_form(w_plus, sigma) 


class FactorRiskCost(BaseCost):
    """PCA Based Factor risk model"""
    def __init__(self, gamma, returns, window_size: int , n_factors: int, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.window_size = window_size
        self.n_factors = n_factors

        self.factor = {}
        self.sigma = {}
        self.idiosync = {}

        self._construct_factor_model(returns)

    def realised_value(self, t, h_plus, u):
        # TO DO
        return 0.

    def _construct_factor_model(self, returns):

        def _pca_factor(d, df):
            second_moments = df.values.T@df.values / self.window_size
            eigen_values, eigen_vectors = np.linalg.eigh(second_moments)
         
            # eigen_values are returned in ascending oarder
            self.factor[d] = eigen_values[-self.n_factors:]
            self.sigma[d] = eigen_vectors[:, -self.n_factors:]

            # residuals
            self.idiosync[d] = eigen_vectors[:, :-self.n_factors]@np.diag(eigen_values[:-self.n_factors])@eigen_vectors[:, :-self.n_factors].T
        
        for d in returns.index[self.window_size:]:
            _pca_factor(d, returns.loc[:d].iloc[-self.window_size:])
                                                      
    def expr(self, t, w_plus, z, v, theta):
        factor = self.factor[t]
        sigma = self.sigma[t]
        idiosync = self.idiosync.loc[t].values

        expression = cvx.sum_squares(cvx.mul_elemwise(np.sqrt(idiosync), w_plus))
        expression += cvx.quad_form(w_plus.T * factor, sigma)
        expression = self.gamma * expression
        return expression

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

    def expr(self, t, w_plus, z, v, theta):
        """Estimate holding cost"""
        w_plus = w_plus[:-1]
        borrow_costs = self.borrow_costs.loc[t].values[:-1]
        dividends = self.dividends.loc[t].values[:-1]
        cost = cvx.mul_elemwise(borrow_costs, cvx.neg(w_plus)) - cvx.mul_elemwise(dividends, w_plus)
        return self.gamma * cvx.sum_entries(cost)

    def realised_value(self, t, h_plus, u):
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

    def expr(self, t, w_plus, z, v, theta):
        """Estimate transaction cost"""
        # TO DO make cash component zero
        z = z[:-1]
        z_abs = cvx.abs(z)
        sigma = self.sigmas.loc[t].values[:-1]
        volumes = self.volumes.loc[t].values[:-1]
        half_spread = self.half_spread.loc[t].values[:-1]
        # this is a n-1 vector
        cost = cvx.mul_elemwise(half_spread, z_abs)
        # v is atom
        # volumes is n-1 vector
        # this stage, it is numpy term, hence auto element wise multiplication
        second_term = (self.nonlin_coef * sigma) * (v / volumes)**(self.nonlin_power-1)      
        
        # now it is cvx element wise multiplication
        cost += cvx.mul_elemwise(second_term, cvx.abs(z)**self.nonlin_power)

        # this is a n-1 vector
        cost += self.asym_coef * z

        return self.gamma * cvx.sum_entries(cost)

    def realised_value(self, t, h_plus, u):
        u = u[:-1]
        u_abs = np.abs(u)
        sigma = self.sigmas.loc[t].values[:-1]
        volumes = self.volumes.loc[t].values[:-1]
        half_spread = self.half_spread.loc[t].values[:-1]
        cost = half_spread * u_abs + \
            self.nonlin_coef * sigma * u_abs**self.nonlin_power / volumes**(self.nonlin_power-1) + \
            self.asym_coef * u
#         print("volume:{}".format(volumes))
#         print("u:{}".format(u))
#         print("cost:{}".format(cost))
        self.last_cost = self.gamma * sum(cost)
        return self.last_cost
