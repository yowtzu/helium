import numpy as np
import pandas as pd

__all__ = ['time_matrix_locator', 'time_locator', 'null_checker', 'non_null_data_args']


def null_checker(obj):
    """Throw exception if obj is not a scalar or pd.DataFrame or contains nulll value"""
    if np.isscalar(obj) and np.isnan(obj):
        raise ValueError('The value of the scalar object is NaN', obj)
    elif isinstance(obj, pd.DataFrame):
        raise ValueError('DataFrame contains NaN', obj)
    else:
        raise TypeError('Data object can only be scalar or pd.DataFrame')
