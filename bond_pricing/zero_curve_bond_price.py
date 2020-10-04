from numpy import (array, arange, empty_like, vectorize,  # noqa E401
                   ceil, log, exp, interp, nan, where, dot,
                   concatenate)
import numpy as np
from scipy.interpolate import CubicSpline
from bond_pricing.simple_bonds import bond_coupon_periods, equiv_rate
from bond_pricing.utils import newton_wrapper, dict_to_dataframe


def par_yld_to_zero(par, freq=1, return_dataframe=False):
    r"""Bootstrap a complete par bond yield curve to zero

    Parameters
    ----------
    par : sequence of floats
          The par bond yields for various maturities in decimal
          Maturities are spaced 1/freq years apart
    freq : int
          The coupon frequency (equals compounding frequency)
    return_dataframe : bool
         whether to return pandas DataFrame instead of dict
    Returns
    -------
    dict or DataFrame:

          zero_yields: array of zero yields in decimal

          zero_prices: array of zero prices

          forward_rates: array of forward rates in decimal

          (Maturities are spaced 1/freq years apart)

    Examples
    --------
    >>> par_yld_to_zero(
    ... par=[1.0200e-2, 1.2000e-2, 1.4200e-2, 1.6400e-2, 1.9150e-2,
    ...      2.1900e-2, 2.4375e-2, 2.6850e-2, 2.9325e-2, 3.1800e-2],
    ... freq=2, return_dataframe=True)
       zero_yields  zero_prices  forward_rates
    0     0.010200     0.994926       0.010200
    1     0.012005     0.988102       0.013812
    2     0.014220     0.978970       0.018656
    3     0.016445     0.967776       0.023133
    4     0.019245     0.953245       0.030487
    5     0.022068     0.936279       0.036242
    6     0.024630     0.917891       0.040066
    7     0.027218     0.897504       0.045429
    8     0.029837     0.875223       0.050915
    9     0.032492     0.851159       0.056545

    """
    annuity = 0
    prev_zero = 1
    zp = empty_like(par)
    zyld = empty_like(par)
    fwd = empty_like(par)
    for i, cpn in enumerate(par):
        n = i + 1
        pv_intermediate = annuity * cpn/freq
        pv_final = 1 - pv_intermediate
        zero_price = pv_final / (1 + cpn/freq)
        zp[i] = zero_price
        zyld[i] = (zero_price**(-1.0/n) - 1) * freq
        fwd[i] = (prev_zero / zero_price - 1) * freq
        prev_zero = zero_price
        annuity += zero_price
    result = dict(zero_yields=zyld, zero_prices=zp, forward_rates=fwd)
    if return_dataframe:
        return dict_to_dataframe(result)
    else:
        return result


def zero_to_par(zero_prices=None, zero_yields=None, freq=1):
    r"""Convert zero yield curve into par curve

    Parameters
    ----------
    zero_prices : sequence of floats
             The zero prices for various maturities in decimal
             Either zero_prices or zero_yields must be provided
             If both are provided, zero_yields is ignored
             Maturities are spaced 1/freq years apart
    zero_yields : sequence of floats
             The zero yields for various maturities in decimal
             Maturities are spaced 1/freq years apart
    freq : int
          The coupon frequency (equals compounding frequency)

    Returns
    -------
    par : array of floats
          The par yields for various maturities in decimal
          Maturities are spaced 1/freq years apart

    Examples
    --------
    >>> zero_to_par(
    ... zero_yields=[0.0102    , 0.0120054 , 0.01421996, 0.01644462,
    ...              0.01924529, 0.02206823, 0.02462961, 0.02721789,
    ...              0.0298373 , 0.0324924 ],
    ... freq=2)  # doctest: +NORMALIZE_WHITESPACE
    array([0.0102  , 0.012   , 0.0142  , 0.0164  , 0.01915 , 0.0219  ,
           0.024375, 0.02685 , 0.029325, 0.0318  ])

    >>> zero_to_par(
    ... zero_prices=[0.99492588, 0.98810183, 0.97896982, 0.96777586,
    ...              0.9532451 , 0.9362787 , 0.91789052, 0.89750426,
    ...              0.87522337, 0.85115892],
    ... freq=2)  # doctest: +NORMALIZE_WHITESPACE
    array([0.0102  , 0.012   , 0.0142  , 0.0164  , 0.01915 , 0.0219  ,
           0.024375, 0.02685 , 0.029325, 0.0318  ])

    """
    if zero_prices is None:
        zero_yields, freq = array(zero_yields), array(freq)
        t = arange(len(zero_yields)) + 1
        zero_prices = (1 + zero_yields/freq)**(-t)
    annuity = 0
    par = empty_like(zero_prices)
    for i, zero_price in enumerate(zero_prices):
        annuity = annuity + zero_price
        pv_redeem = zero_price
        pv_cpns = 1 - pv_redeem
        par[i] = (pv_cpns / annuity) * freq
    return par


def nelson_siegel_zero_rate(beta0, beta1, beta2, tau, m):
    r"""Computes zero rate from Nelson Siegel parameters

    Parameters
    ----------
    beta0 : float
            the long term rate
    beta1 : float
            the slope
    beta2 : float
            curvature
    tau : float
            location of the hump
    m : float
            maturity at which zero rate is to be found

    Returns
    -------
    float :
        the continuously compounded zero yield for maturity m

    Examples
    --------
    >>> nelson_siegel_zero_rate(0.128397, -0.024715, -0.050231, 2.0202,
    ...                         [0.25, 5, 15, 30])
    array([0.10228692, 0.10489195, 0.11833924, 0.12335016])

    >>> nelson_siegel_zero_rate(0.0893088, -0.0314768, -0.0130352,
    ...                         3.51166, [0.25, 5, 15, 30])
    array([0.05848376, 0.06871299, 0.07921554, 0.08410199])

    """
    m = array(m)
    old_settings = np.seterr(invalid='ignore')
    possible_0_by_0 = where(m == 0,
                            1,
                            np.divide(1 - exp(-m/tau), m/tau))
    np.seterr(**old_settings)
    return (beta0 + (beta1 + beta2) * possible_0_by_0
            - beta2 * exp(-m/tau))[()]


def make_zero_price_fun(flat_curve=None,
                        nelson_siegel=None,
                        par_at_knots=None,
                        par_at_coupon_dates=None,
                        zero_at_knots=None,
                        zero_at_coupon_dates=None,
                        freq=1, max_maturity=None):
    r"""Create function that returns zero price for any maturity

    Parameters
    ----------
    flat_curve : float
           Yield (flat yield curve)
    nelson_siegel : tuple of floats
           tuple consists of beta0, beta1, beta2, tau
    par_at_knots : tuple of two sequences of floats
           First element of tuple is a sequence of maturities
           Second element is a sequence of par yields for these maturities
    par_at_coupon_dates : sequence of floats
           Par yields for maturities spaced 1/freq years apart
    zero_at_knots : tuple of two sequences of floats
           First element of tuple is a sequence of maturities
           Second element is a sequence of zero rates for these maturities
    zero_at_coupon_dates : sequence
          Zero yields for maturities spaced 1/freq years apart
    freq : int
          The coupon frequency (equals compounding frequency)
    max_maturity : float
          The maximum maturity upto which the yields are to be
          extrapolated. If None, no extrapolation is done.

    Returns
    -------
    function :
          Function that takes float (maturity) as argument and
          returns float (zero price)

    Examples
    --------
    >>> make_zero_price_fun(par_at_knots = (
    ... [0, 1, 3, 5, 10],
    ... [3e-2, 3.5e-2,  4e-2, 4.5e-2, 4.75e-2]))([1, 5, 10])
    array([0.96618357, 0.80065643, 0.62785397])

    """
    if flat_curve is not None:
        return lambda t: exp(-array(t) * log(1 + flat_curve))[()]
    if nelson_siegel is not None:
        beta0, beta1, beta2, tau = nelson_siegel
        return lambda m: exp(- nelson_siegel_zero_rate(
            beta0, beta1, beta2, tau, m) * array(m))[()]
    if par_at_knots is not None:
        t, r = par_at_knots
        if min(t) != 0:
            from warnings import warn
            if min(t) < 0:
                warn("Knot point at negative maturity.")
            else:
                warn("A knot point at zero maturity is recommended.")
        if max_maturity is None:
            max_maturity = max(t)
        par_at_coupon_dates = CubicSpline(t, r)(
            arange(ceil(max_maturity*freq))+1)
    if par_at_coupon_dates is not None:
        zero_at_coupon_dates = par_yld_to_zero(par_at_coupon_dates,
                                               freq)['zero_yields']
    if zero_at_coupon_dates is None:
        assert zero_at_knots is not None, "No yield curve provided"
        t, r = zero_at_knots
        if min(t) != 0:
            from warnings import warn
            if min(t) < 0:
                warn("Knot point at negative maturity.")
            else:
                warn("A knot point at zero maturity is recommended.")
        if max_maturity is None:
            max_maturity = max(t)
        zero_at_coupon_dates = CubicSpline(t, r)(
            arange(ceil(max_maturity*freq))+1)
    zero_at_coupon_dates = array(zero_at_coupon_dates)
    t = arange(len(zero_at_coupon_dates) + 1) / freq
    log_zero_df = -log(concatenate([[1], 1 + zero_at_coupon_dates])) * t
    return lambda x: exp(interp(x, t, log_zero_df, left=nan, right=nan))


def zero_curve_bond_price_breakup(
        settle=None, cpn=0, mat=1, zero_price_fn=(lambda x: 1),
        freq=2, face=100, redeem=None, daycount=None,
        return_dataframe=False):
    r"""Clean/dirty price & accrued interest of coupon bond using zero yields

    Parameters
    ----------
    settle : date or None
          The settlement date. None means maturity is in years.
    cpn : float
          The coupon rate in decimal
    mat : float or date
          Maturity date or if settle is None, maturity in years
    zero_price_fn : function
          takes float (maturity) as argument and
          returns float (zero price)
    freq : int
          Coupon frequency
    face : float
          Face value of the bond
    redeem : float
          Redemption value
    daycount : daycounter
          This class has day_count and year_fraction methods
    return_dataframe : bool
         whether to return pandas DataFrame instead of dict

    Returns
    -------
    dict or DataFrame
         dirty: dirty price of the bond

         accrued: accrued interest

         clean: clean price of the bond

         next_coupon: Next coupon date (only if settle is None)

         prev_coupon: Previous coupon date (only if settle is None)

    Examples
    --------
    >>> zero_curve_bond_price_breakup(
    ...     cpn=10e-2, mat=10, freq=1,
    ...     zero_price_fn=make_zero_price_fun(
    ...         flat_curve=8e-2))  # doctest: +NORMALIZE_WHITESPACE
    {'DirtyPrice': 113.42016279788285,
     'AccruedInterest': 0.0,
     'CleanPrice': 113.42016279788285,
     'NextCoupon': None,
     'PreviousCoupon': None}

    >>> zero_curve_bond_price_breakup(
    ...     cpn=5e-2, mat=2, freq=1,
    ...     zero_price_fn=make_zero_price_fun(
    ...         zero_at_coupon_dates=[3e-2, 10e-2])
    ... )  # doctest: +NORMALIZE_WHITESPACE
    {'DirtyPrice': 91.63122843617106,
     'AccruedInterest': 0.0,
     'CleanPrice': 91.63122843617106,
     'NextCoupon': None,
     'PreviousCoupon': None}

    """
    freq, cpn = array(freq), array(cpn)
    redeem = where(redeem is None, face, redeem)
    red_by_face = redeem / face
    res = bond_coupon_periods(settle, mat, freq, daycount)
    # print(res)

    def one_dirty_price(n, fraction, coupon, R_by_F, zp_fun):
        t = (arange(n) + fraction) / freq
        df = zp_fun(t)
        cf = [coupon/freq] * int(n-1) + [coupon/freq + R_by_F]
        # import pandas as pd
        # print(pd.DataFrame(dict(df=df, cf=cf), index=t))
        return dot(df, cf)

    dirty_price = vectorize(one_dirty_price)
    dirty = dirty_price(n=res['n'],
                        fraction=res['discounting_fraction'],
                        coupon=cpn,
                        R_by_F=red_by_face,
                        zp_fun=zero_price_fn)
    accrued = cpn/freq * res['accrual_fraction']
    clean = dirty - accrued
    result = dict(DirtyPrice=(face*dirty)[()],
                  AccruedInterest=(face*accrued)[()],
                  CleanPrice=(face*clean)[()],
                  NextCoupon=res['next_coupon'],
                  PreviousCoupon=res['prev_coupon'])
    if return_dataframe:
        return dict_to_dataframe(result)
    else:
        return result


def zero_curve_bond_price(settle=None, cpn=0, mat=1,
                          zero_price_fn=(lambda x: 1),
                          freq=2, face=100, redeem=None,
                          daycount=None):
    r"""Compute clean price of coupon bond using zero yields

    Parameters
    ----------
    settle : date or None
          The settlement date. None means maturity is in years.
    cpn : float
          The coupon rate in decimal
    mat : float or date
          Maturity date or if settle is None, maturity in years
    zero_price_fn : function
          takes float (maturity) as argument and
          returns float (zero price)
    freq : int
          Coupon frequency
    face : float
          Face value of the bond
    redeem : float
          Redemption value
    daycount : daycounter
          This class has day_count and year_fraction methods

    Returns
    -------
    float
         clean price of the bond

    Examples
    --------
    >>> zero_curve_bond_price(
    ...     cpn=10e-2, mat=10, freq=1,
    ...     zero_price_fn=make_zero_price_fun(
    ...         flat_curve=8e-2))  # doctest: +NORMALIZE_WHITESPACE
    113.42016279788285

    >>> zero_curve_bond_price(
    ...     cpn=5e-2, mat=2, freq=1,
    ...     zero_price_fn=make_zero_price_fun(
    ...         zero_at_coupon_dates=[3e-2, 10e-2])
    ... )  # doctest: +NORMALIZE_WHITESPACE
    91.63122843617106

    >>> zero_curve_bond_price(
    ...     cpn=5.792982e-2, mat=6, freq=2,
    ...     zero_price_fn=make_zero_price_fun(
    ...         nelson_siegel=(6.784e-2, -3.8264e-2, -3.6631e-2, 0.7774))
    ... )
    99.99999939355965

    """
    return zero_curve_bond_price_breakup(
        settle=settle, cpn=cpn, mat=mat, zero_price_fn=zero_price_fn,
        freq=freq, face=face, redeem=redeem,
        daycount=daycount)['CleanPrice']


def zero_curve_bond_duration(settle=None, cpn=0, mat=1,
                             zero_price_fn=(lambda x: 1),
                             freq=2, face=100, redeem=None,
                             daycount=None, modified=False):
    r"""Duration of coupon bond using zero yields

    Parameters
    ----------
    settle : date or None
          The settlement date. None means maturity is in years.
    cpn : float
          The coupon rate in decimal
    mat : float or date
          Maturity date or if settle is None, maturity in years
    zero_price_fn : function
          takes float (maturity) as argument and
          returns float (zero price)
    freq : int
          Coupon frequency
    face : float
          Face value of the bond
    redeem : float
          Redemption value
    daycount : daycounter
          This class has day_count and year_fraction methods

    Returns
    -------
    float
         duration or modified duration of the bond

    Examples
    --------
    >>> zero_curve_bond_duration(
    ...     cpn=10e-2, mat=10, freq=1,
    ...     zero_price_fn=make_zero_price_fun(
    ...         flat_curve=8e-2))  # doctest: +NORMALIZE_WHITESPACE
    6.965803939497351
    >>> zero_curve_bond_duration(
    ...     cpn=5e-2, mat=2, freq=1,
    ...     zero_price_fn=make_zero_price_fun(
    ...         zero_at_coupon_dates=[3e-2, 10e-2])
    ... )  # doctest: +NORMALIZE_WHITESPACE
    1.9470227670753064
    """
    freq, cpn = array(freq), array(cpn)
    redeem = where(redeem is None, face, redeem)
    red_by_face = redeem / face
    res = bond_coupon_periods(settle, mat, freq, daycount)

    def one_duration(n, fraction, coupon, R_by_F, zp_fun):
        t = (arange(n) + fraction) / freq
        df = zp_fun(t)
        cf = [coupon/freq] * int(n-1) + [coupon/freq + R_by_F]
        # import pandas as pd
        # print(pd.DataFrame(dict(df=df, cf=cf), index=t))
        return dot(df, cf * t) / dot(df, cf)

    duration = vectorize(one_duration)
    result = duration(n=res['n'],
                      fraction=res['discounting_fraction'],
                      coupon=cpn,
                      R_by_F=red_by_face,
                      zp_fun=zero_price_fn)
    return result[()]


def static_zero_spread(settle=None, cpn=0, mat=1, price=None,
                       zero_price_fn=(lambda x: 1),
                       freq=2, face=100, redeem=None,
                       daycount=None, guess=0.0):
    r"""Static spread (Z-spread) over zero curve to match bond price

    Parameters
    ----------
    settle : date or None
          The settlement date. None means maturity is in years.
    cpn : float
          The coupon rate in decimal
    mat : float or date
          Maturity date or if settle is None, maturity in years
    price : float
          Market price of the bond
    zero_price_fn : function
          takes float (maturity) as argument and
          returns float (zero price)
    freq : int
          Coupon frequency
    face : float
          Face value of the bond
    redeem : float
          Redemption value
    daycount : daycounter
          This class has day_count and year_fraction methods
    guess : float
          Initial guess of the yield for the root finder

    Returns
    -------
    float
         Static spread (Z-spread)

    Examples
    --------
    >>> round(static_zero_spread(
    ...     cpn=5e-2, mat=2, freq=1, price=91.63122843617106,
    ...     zero_price_fn=make_zero_price_fun(
    ...         zero_at_coupon_dates=[2.5e-2, 9.5e-2])),
    ... 6)
    0.005
    """

    def zero_price_fn_with_spread(spread, freq):
        def f(t):
            cc_yld = -log(zero_price_fn(t)) / t
            new_yld = equiv_rate(cc_yld, np.inf, freq) + spread
            new_cc_yld = equiv_rate(new_yld, freq, np.inf)
            return exp(-new_cc_yld * t)
        return f

    def one_spread(settle, cpn, mat, price, zero_price_fn,
                   freq, face, redeem, daycount, guess):
        return newton_wrapper(
            lambda spread: zero_curve_bond_price(
                settle=settle, cpn=cpn, mat=mat,
                zero_price_fn=zero_price_fn_with_spread(spread, freq),
                freq=freq, face=face,
                redeem=redeem, daycount=daycount) - price,
            guess)
    spread = vectorize(one_spread)
    result = spread(settle=settle, cpn=cpn, mat=mat, price=price,
                    zero_price_fn=zero_price_fn, freq=freq, face=face,
                    redeem=redeem, daycount=daycount, guess=guess)
    return result[()]
