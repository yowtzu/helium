from abc import ABCMeta, abstractmethod


class Expression(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def w_expr(self, t, w_post, z, value):
        pass

    def w_hat_expr(self, t, tau, w_post, z, value):
        """Return the estimate at time to of cost at time tau"""
        return self.weight_expr(t, w_plus, z, value)

    def value_expr(self, t, h_plus, u):
        pass
