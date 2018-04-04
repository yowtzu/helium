from abc import ABCMeta, abstractmethod

class BaseRisk(object):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.w_benchmark = kwargs.pop('w_benchmark', 0.)
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
            
class FactorModelRisk(BaseRisk):
    
    def __init__(self, returns, **kwargs):
        super(FactorModelRisk, self).__init__(**kwargs)
        self._construct_factor_model(returns)

    def _construct_factor_model(self, returns, window_size, n_factors, **kwargs):
        self.window_size = window_size
        self.n_factor = n_factors
        self.factor = {}
        self.sigma = {}
        self.idiosync = {}
        
        def pca_factor(df, k):
            t = df.index[-1]
            second_moments = df.values.T@df.values / df.values.shape[0]
            eigen_values, eigen_vectors = np.linalg.eigh(second_moments)
            # eigen_values are returned in ascending order
            self.factor[t] = eigen_values[-k:]
            self.sigma[t] = eigen_vectors[:, -k:]
            # residuals
            self.idiosync[t] = eigen_vectors[:,:-k]@np.diag(eigen_values[:-k])@eigen_vectors[:,:-k].T
        
        for d in range(window_size, len(returns)):
            pca_factor(returns.loc[d-window_size:d], n_factor)        
          
    def _estimate(self, t, wplus, z, v):
        factor_risk = self.factor_risk[t]
        factor_loading = self.factor_loading[t]
        idiosync = self.idiosync.loc[t].values
        cvx.quad_form(wplus.T * factor_loading, * factor_risk)
