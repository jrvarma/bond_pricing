"""Utility functions

`newton_wrapper` is a wrapper for scipy.optimize.newton to return root or nan

`edate` moves date(s) by specified no of months (similar to excel edate)

`dataframe_like_dict` creates pandas DataFrame from dict like arguments

"""
from scipy.optimize import newton
import pandas as pd
import numpy as np


def newton_wrapper(f, guess, warn=True):
    r"""Wrapper for `scipy.optimize.newton` to return root or `nan`

    Parameters
    ----------
    f : callable
        The function whose zero is to be found
    guess : float
        An initial estimate of the zero somewhere near the actual zero.
    warn : bool, Optional
        If true, a warning is issued when returning nan.
        This happens when `scipy.optimize.newton` does not converge.

    Returns
    -------
    float
         The root if found, numpy.nan otherwise.
    Examples
    --------
    >>> newton_wrapper(lambda x: x**2 - x, 0.8)
    1.0
    >>> newton_wrapper(lambda x: x**2 + 1, 1, warn=False)
    nan
    """
    root, status = newton(f, guess, full_output=True, disp=False)
    if status.converged:
        return root
    else:
        if warn:
            from warnings import warn
            warn("Newton root finder did not converge. Returning nan")
        return np.nan


def _edate_0(dt, m):
    """Unvectorized form of edate

    Please see its documentation

    Examples
    --------

    >>> _edate_0('2020-01-31', 1)
    Timestamp('2020-02-29 00:00:00')
    """
    return pd.to_datetime(dt) + pd.tseries.offsets.DateOffset(months=m)


_edate = np.vectorize(_edate_0)


def edate(dt, m):
    """Move date(s) by specified no of months (similar to excel edate)

    Parameters
    ----------
    dt : date, object convertible to date or sequence
         The date(s) to be shifted
    m : int or sequence
         Number of months to shift by

    Returns
    -------
    date
        dt shifted by m months

    Examples
    --------

    >>> edate('2020-01-31', 1)
    Timestamp('2020-02-29 00:00:00')


    >>> edate(['2020-01-31', '2020-03-31'], [1, 6])
    array([Timestamp('2020-02-29 00:00:00'), Timestamp('2020-09-30 00:00:00')],
          dtype=object)
    """
    return _edate(dt, np.array(m))[()]


def dict_to_dataframe(d):
    lengths = [1]
    lengths += [len(x) for x in d.values()
                if hasattr(x, "len")
                and not isinstance(x, str)]
    lengths += [x.shape[0] for x in d.values()
                if hasattr(x, "shape")
                and len(x.shape) > 0]
    n = max(lengths)
    return pd.DataFrame(d, index=range(n))
