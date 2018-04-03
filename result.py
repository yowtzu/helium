import collections
import numpy as np
import pandas as pd
import copy

class SimulationResult():
    """A container to store simulation result

    Attr:
        h_next: A dataframe of holdings over time
    """

    def __init__(self):
        pass

    def summary(self):
        print(self._summary_info())

    def _summary_info(self):
        data = collections.OrderedDict({})
        return pd.Series()

    @property
    def excess_returns(self):
        return self.returns - self.risk_free_returns

    @property
    def sharpe_ratio(self):
        return self.ann_factor * self.excess_returns.mean() / self.excess_returns.std()


