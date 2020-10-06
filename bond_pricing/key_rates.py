import numpy as np
import pandas as pd
from bond_pricing.zero_curve_bond_price import (  # noqa F401
    zero_to_par, par_yld_to_zero, make_zero_price_fun,
    zero_curve_bond_price)


standard_krs_points = [2, 5, 10, 30]


def _mat_sequence(T, freq):
    return np.arange(1/freq, T + 1/freq, 1/freq)


def key_rate_shift(krs, mat=None, T=None, freq=2,
                   krs_points=standard_krs_points):
    r"""Shift in various par bond yields for a key rate shift

    Mainly for internal use or pedagogical use.


    Parameters
    ----------
    krs : int or string, optional
          The key rate segment of the curve to be shifted
          For example, 5 for shifting the 5 year segment.
          "all" and "none" are also accepted.
    mat : array
          The maturities for which the shift is to be computed
    T : Maximum maturity upto which shifted yields are to be returned.
    freq : Coupon or Compounding Frequency
    krs_points : sequence, optional
           The key rates to be used.

    Returns
    -------
    array
      The shifts in each par bond yields for a one basis point
      shift in a key rate

    Examples
    --------

    """
    if mat is None:
        if T is None:
            T = krs_points[-1]
        mat = _mat_sequence(T, freq)
    if krs in krs_points:
        bump = [0] * len(krs_points)
        bump[krs_points.index(krs)] = 1
        return np.interp(mat, krs_points, bump)
    elif krs == "all":
        return np.ones_like(mat)
    elif krs == "none":
        return np.zeros_like(mat)
    else:
        raise(Exception("Invalid key rate shift"))


def make_KRS(freq=2, krs_points=standard_krs_points):
    r"""Data Frame of par yield shifts for all key rate shifts

    Mainly for pedagogical use.

    Parameters
    ----------
    freq : Coupon or Compounding Frequency
    krs_points : sequence, optional
           The key rates to be used.

    Returns
    -------
    pandas DataFrame

    Examples
    --------

    """
    KRS = pd.DataFrame(index=_mat_sequence(krs_points[-1], freq))
    for krs in list(krs_points) + ['all', 'none']:
        KRS[krs] = key_rate_shift(krs, freq=freq, krs_points=krs_points)
    return KRS


def key_rate_shifted_zero_curve(
        initial_zero_fn=None, initial_zero_price=None,
        initial_zero_yld=None, initial_par_yld=None,
        krs='none', freq=2, T=None, what='zero_yields',
        krs_points=standard_krs_points):
    r"""Applies a key rate shift to an yield curve

    The initial yield curve can be given in many alternative ways.
    One of these must be non None:
    `initial_zero_fn`, `initial_zero_price`, `initial_zero_yld`,
    `initial_par_yld`

    Parameters
    ----------
    initial_zero_fn : function, optional
                      Function that returns zero prices for any maturity
    initial_zero_price : array, optional
                      Zero prices for coupon dates up to some maturity
    initial_zero_yld : array, optional
                      Zero yields for coupon dates up to some maturity
    initial_par_yld : array, optional
                      Par yields for coupon dates up to some maturity
    krs : int or string, optional
          The key rate segment of the curve to be shifted
          For example, 5 for shifting the 5 year segment.
          "all" and "none" are also accepted.
    freq : Coupon or Compounding Frequency
    T : Maximum maturity upto which shifted yields are to be returned.
    what : str, optional
           What to return. Can be:
           `zero_yields`, `zero_prices`, `forward_rates` or `zero_fn`
    krs_points : sequence, optional
           The key rates to be used.
    Returns
    -------
    array or function:
       If `what` is `zero_yields`, `zero_prices` or `forward_rates`,
       the corresponding array is return. If `what` is `zero_fn`, a
       function is returned.

    Examples
    --------
    Compute KR01 of a 10-year 8% bond when initial yield curve is
    flat at 5%. Being a 10 year bond, the 10 year KR01 is the largest.
    But since it is not a par bond, the 5 year KR01 is also large.

    >>> fn0 = make_zero_price_fun(flat_curve=5e-2, freq=2)
    >>> P0 = zero_curve_bond_price(cpn=8e-2, mat=10, zero_price_fn=fn0)
    >>> fns = [key_rate_shifted_zero_curve(
    ...        initial_zero_fn=fn0, krs=krs, what='zero_fn')
    ...        for krs in standard_krs_points]
    >>> zero_curve_bond_price(cpn=8e-2, mat=10, zero_price_fn=fns) - P0
    array([-0.00121608, -0.00467591, -0.08307928,  0.        ])

    """
    if initial_zero_fn is not None:
        if T is None:
            T = krs_points[-1]
        initial_zero_price = initial_zero_fn(_mat_sequence(T, freq))
    if initial_zero_price is not None:
        initial_par_yld = zero_to_par(zero_prices=initial_zero_price,
                                      freq=freq)
    elif initial_zero_yld is not None:
        initial_par_yld = zero_to_par(zero_yields=initial_zero_yld,
                                      freq=freq)
    assert initial_par_yld is not None, (
        "zero_fn, zero_yld, zero_price or par_yld must be given")
    if not isinstance(initial_par_yld, (np.ndarray, pd.Series)):
        initial_par_yld = np.asarray(initial_par_yld)
    shifted_par = initial_par_yld + key_rate_shift(
        krs, T=len(initial_par_yld)/freq, freq=freq,
        krs_points=krs_points) / 10000
    if what in "zero_yields zero_prices forward_rates".split():
        return par_yld_to_zero(shifted_par, freq=freq)[what]
    elif what == "zero_fn":
        zyld = par_yld_to_zero(shifted_par, freq=freq)['zero_yields']
        return make_zero_price_fun(zero_at_coupon_dates=zyld, freq=freq)
    else:
        raise(Exception("'what' must be one of zero_yields zero_prices "
                        "forward_rates zero_fn"))
