"""Bond Valuation functions

Important functions are `bond_price`, `bond_yield` and `bond_duration`
for computing the price, yield to maturity and duration of one
or more bonds.

The settlement date and maturity date can be given as dates and the
software calculates the time to maturity (and coupon dates) using a
`daycount` convention to calculate year_fractions.
For any `daycount` other than simple counting of days
(ACT/365 in ISDA terminology), this packages relies on the
`isda_daycounters` module that can be downloaded from
<https://github.com/miradulo/isda_daycounters>

Maturity can alternatively be given in years by setting `settle`
to `None`. The trade/settlement date is then year 0.

"""
from numpy import array, where, vectorize, float64, ceil
import pandas as pd
from bond_pricing.utils import (
    newton_wrapper, edate, dict_to_dataframe)
from bond_pricing.present_value import pvaf, equiv_rate


def _bond_coupon_periods_0(settle=None, mat=1, freq=2, daycount=None):
    r"""Unvectorized form of bond_coupon_periods.

    Please see its documentation

    """
    if settle is not None:
        settle = pd.to_datetime(settle)
        mature = pd.to_datetime(mat)
        # approximate number of full coupon periods left
        n = int(freq * (mature - settle).days / 360)
        # the divisor of 360 guarantees this is an overestimate
        # we keep reducing n till it is right
        while(edate(mature, -n * 12 / freq) <= settle):
            n -= 1
        next_coupon = edate(mature, -n * 12 / freq)
        prev_coupon = edate(next_coupon, -12/freq)
        if prev_coupon != settle:
            # we are in the middle of a coupon period
            # no of coupons is One PLUS No of full coupon periods
            n += 1
        if daycount is None:
            daycounter = default_daycounter
        else:
            assert daycount in daycounters, (
                "Unknown daycount {:}. {:}".format(
                    daycount,
                    "isda_daycounters not available"
                    if no_isda else ""))
            daycounter = daycounters[daycount]
        discounting_fraction = daycounter.year_fraction(
            settle, next_coupon) * freq
        accrual_fraction = daycounter.year_fraction(
            prev_coupon, settle) * freq
    else:
        n = ceil(mat*freq)
        accrual_fraction = n - freq * mat
        discounting_fraction = 1 - accrual_fraction
        next_coupon = None
        prev_coupon = None
    if accrual_fraction == 1:
        # We are on coupon date. Assume that bond is ex-interest
        # Remove today's coupon
        # This affects dirty price and accrued interest
        # but not clean price
        discounting_fraction += 1
        accrual_fraction -= 1
    return (n, discounting_fraction, accrual_fraction,
            next_coupon, prev_coupon)


_bond_coupon_periods = vectorize(_bond_coupon_periods_0)


def bond_coupon_periods(settle=None, mat=1, freq=2, daycount=None,
                        return_dataframe=False):
    r""" Compute no of coupon, coupon dates and fractions

    Parameters
    ----------
    settle : date or None
          The settlement date. None means maturity is in years.
    mat : int or date
          Maturity date or if settle is None, maturity in years
    freq : int
          Coupon frequency
    daycount : daycounter
          This class has day_count and year_fraction methods
    return_dataframe : bool
         whether to return pandas DataFrame instead of dict

    Returns
    -------
    dict
        n : No of coupons left

        discounting_fraction: Fraction of coupon period to next coupon

        accrual_fraction: Fraction of coupon period to previous coupon

        next_coupon: Next coupon date (None if settle is None)

        prev_coupon Previous coupon date (None if settle is None)

    Examples
    --------
    >>> bond_coupon_periods(
    ... settle='2020-03-13', mat='2030-01-01', freq=2, daycount=None
    ... )  # doctest: +NORMALIZE_WHITESPACE
    {'n': 20,
    'discounting_fraction': 0.6,
    'accrual_fraction': 0.4,
    'next_coupon': Timestamp('2020-07-01 00:00:00'),
    'prev_coupon': Timestamp('2020-01-01 00:00:00')}

    >>> bond_coupon_periods(
    ... mat=10.125, freq=2, daycount=None
    ... )  # doctest: +NORMALIZE_WHITESPACE
    {'n': 21.0,
    'discounting_fraction': 0.25,
    'accrual_fraction': 0.75,
    'next_coupon': None,
    'prev_coupon': None}

    """
    res = array(_bond_coupon_periods(settle=settle, mat=mat,
                                     freq=freq, daycount=daycount))
    if len(res.shape) > 1:
        result = dict(n=res[0, :],
                      discounting_fraction=res[1, :],
                      accrual_fraction=res[2, :],
                      next_coupon=res[3, :],
                      prev_coupon=res[4, :],)
    else:
        result = dict(n=res[0][()],
                      discounting_fraction=res[1][()],
                      accrual_fraction=res[2][()],
                      next_coupon=res[3][()],
                      prev_coupon=res[4][()],)
    if return_dataframe:
        return dict_to_dataframe(result)
    else:
        return result


def bond_price_breakup(settle=None, cpn=0, mat=1, yld=0, freq=2,
                       comp_freq=None, face=100, redeem=None,
                       daycount=None, return_dataframe=False):
    r"""Compute clean/dirty price & accrued_interest of coupon bond using YTM

    Parameters
    ----------
    settle : date or None
          The settlement date. None means maturity is in years.
    cpn : float
          The coupon rate in decimal
    mat : float or date
          Maturity date or if settle is None, maturity in years
    yld : float
          The yield to maturity in decimal
    freq : int
          Coupon frequency
    comp_freq : int
          Compounding frequency
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
    dict
         dirty: dirty price of the bond

         accrued: accrued interest

         clean: clean price of the bond

         next_coupon: Next coupon date (only if settle is None)

         prev_coupon: Previous coupon date (only if settle is None)

    Examples
    --------
    >>> bond_price_breakup(
    ... settle="2012-04-15", mat="2022-01-01", cpn=8e-2, yld=8.8843e-2,
    ... freq=1)  # doctest: +NORMALIZE_WHITESPACE
    {'DirtyPrice': 96.64322827099208,
    'AccruedInterest': 2.311111111111111,
    'CleanPrice': 94.33211715988098,
    'NextCoupon': Timestamp('2013-01-01 00:00:00'),
    'PreviousCoupon': Timestamp('2012-01-01 00:00:00')}

    >>> bond_price_breakup(
    ... mat=10.25, cpn=8e-2, yld=9e-2,
    ... freq=2)  # doctest: +NORMALIZE_WHITESPACE
    {'DirtyPrice': 95.37373582338677,
    'AccruedInterest': 2.0,
    'CleanPrice': 93.37373582338677,
    'NextCoupon': None,
    'PreviousCoupon': None}

    >>> bond_price_breakup(
    ... settle="2012-04-15", mat="2022-01-01", cpn=8e-2, yld=8.8843e-2,
    ... freq=[1,2,4])  # doctest: +NORMALIZE_WHITESPACE
    {'DirtyPrice': array([96.64322827099208, 96.61560601490777,
     94.59488828646788], dtype=object),
    'AccruedInterest': array([2.311111111111111, 2.311111111111111,
    0.3111111111111111], dtype=object),
    'CleanPrice': array([94.33211715988098, 94.30449490379667,
    94.28377717535678], dtype=object),
    'NextCoupon': array([Timestamp('2013-01-01 00:00:00'),
                        Timestamp('2012-07-01 00:00:00'),
                        Timestamp('2012-07-01 00:00:00')],
    dtype=object),
    'PreviousCoupon': array([Timestamp('2012-01-01 00:00:00'),
                            Timestamp('2012-01-01 00:00:00'),
                            Timestamp('2012-04-01 00:00:00')],
                            dtype=object)}
    >>> bond_price_breakup(
    ... settle="2012-04-15", mat="2022-01-01", cpn=8e-2, yld=8.8843e-2,
    ... freq=[1,2,4],
    ... return_dataframe=True)
      DirtyPrice AccruedInterest CleanPrice NextCoupon PreviousCoupon
    0    96.6432         2.31111    94.3321 2013-01-01     2012-01-01
    1    96.6156         2.31111    94.3045 2012-07-01     2012-01-01
    2    94.5949        0.311111    94.2838 2012-07-01     2012-04-01

    """
    # None can make comp_freq an array of objects
    # Since log needs float, we use .astype(float64)
    freq, cpn = array(freq), array(cpn)
    comp_freq = where(comp_freq is None, freq, comp_freq).astype(float64)
    # find the equivalent yield that matches the coupon frequency
    yld = equiv_rate(yld, comp_freq, freq)
    redeem = where(redeem is None, face, redeem)
    red_by_face = redeem / face
    res = bond_coupon_periods(settle, mat, freq, daycount)
    # compounding factor from previous coupon date to settlement date
    fractional_CF = (1 + yld/freq)**res['accrual_fraction']
    # calculate PV as at previous coupon date and compound to today
    dirty = fractional_CF * (
        # PV of coupons as at previous coupon date
        cpn/freq * pvaf(yld/freq, res['n'])
        # PV of redemption as at previous coupon date
        + red_by_face * (1 + yld/freq)**-res['n'])
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


def bond_price(settle=None, cpn=0, mat=1,
               yld=0, freq=2, comp_freq=None,
               face=100, redeem=None, daycount=None):
    """ Compute clean price of coupon bond using YTM

    This is a wrapper for bond_price_breakup that extracts
    and returns only the clean price.

    Parameters
    ----------
    settle : date or None
          The settlement date. None means maturity is in years.
    cpn : float
          The coupon rate in decimal
    mat : float or date
          Maturity date or if settle is None, maturity in years
    yld : float
          The yield to maturity in decimal
    freq : int
          Coupon frequency
    comp_freq : int
          Compounding frequency
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
    >>> bond_price(settle="2012-04-15", mat="2022-01-01", cpn=8e-2,
    ... yld=8.8843e-2, freq=1)
    94.33211715988098

    >>> bond_price(mat=10.25, cpn=8e-2, yld=9e-2, freq=2)
    93.37373582338677

    >>> bond_price(settle="2012-04-15", mat="2022-01-01", cpn=8e-2,
    ...            yld=8.8843e-2,freq=[1, 2, 4])
    array([94.33211715988098, 94.30449490379667, 94.28377717535678],
          dtype=object)

    """
    return bond_price_breakup(
        settle=settle, cpn=cpn, mat=mat, yld=yld, freq=freq,
        comp_freq=comp_freq, face=face, redeem=redeem,
        daycount=daycount)['CleanPrice']


def bond_duration(settle=None, cpn=0, mat=1, yld=0, freq=2,
                  comp_freq=None, face=100, redeem=None,
                  daycount=None, modified=False):
    r"""

    Parameters
    ----------
    settle : date or None
          The settlement date. None means maturity is in years.
    cpn : float
          The coupon rate in decimal
    mat : float or date
          Maturity date or if settle is None, maturity in years
    yld : float
          The yield to maturity in decimal
    freq : int
          Coupon frequency
    comp_freq : int
          Compounding frequency
    face : float
          Face value of the bond
    redeem : float
          Redemption value
    daycount : daycounter
          This class has day_count and year_fraction methods
    modified : bool
          Whether to return modified duration

    Returns
    -------
    float
         The duration (modified duration if modified is True)

    Examples
    --------
    >>> bond_duration(settle="2012-04-15", mat="2022-01-01", cpn=8e-2,
    ...               yld=8.8843e-2)
    6.678708669753968

    >>> bond_duration(settle="2012-04-15", mat="2022-01-01", cpn=8e-2,
    ...               yld=8.8843e-2, modified=True)
    6.394648779016871

    >>> bond_duration(settle="2012-04-15", mat="2022-01-01", cpn=8e-2,
    ...               yld=[7e-2, 8.8843e-2])
    array([6.88872548, 6.67870867])

    """
    # consider a bond which has just paid a coupon
    # decompose it as a portfolio of three positions
    # Position                 PV           Duration_minus_(1+y)/y
    # --------                 --           ----------------------
    # perpetuity of c       c/y = cF/Fy                0
    # short perpetuity     -c/y/F = -c/Fy              T
    # Redemption of R        R/F = Ry/Fy            T-(1+y)/y
    # where F is Future Value Factor (1+y)^T
    # Portfolio Duration is (1+y)/y - N/D where
    # N = cT - RyT + R(1+y) = R[1+y + T(c/R-y)]
    # D = cF - c + Ry = R[c(F-1)/R + y]
    # where Fy has been eliminated from both N and D
    # Eliminating R gives
    # N = 1+y + T(c/R-y)
    # D = c(F-1)/R + y
    # we compute duration in coupon periods as on previous coupon date
    freq, cpn = array(freq), array(cpn)
    comp_freq = where(comp_freq is None, freq, comp_freq).astype(float64)
    # find the equivalent yield that matches the coupon frequency
    yld = equiv_rate(yld, comp_freq, freq)
    redeem = where(redeem is None, face, redeem)
    R = redeem / face
    res = bond_coupon_periods(settle=settle, mat=mat, freq=freq,
                              daycount=daycount)
    y = yld/freq
    c = cpn/freq
    T = res['n']
    # T += where(res['accrual_fraction'] > 0, 1, 0)
    F = (1+y)**T
    dur = (1+y)/y - (1+y + T*(c/R-y)) / (c*(F-1)/R + y)
    # now we subtract the fractional coupon period
    dur -= res['accrual_fraction']
    # then we convert to years
    dur /= freq
    return where(modified, dur/(1+y), dur)[()]


def _bond_yield_0(settle=None, cpn=0, mat=1, price=100, freq=2, comp_freq=None,
                  face=100, redeem=None, daycount=None, guess=10e-2):
    r"""Unvectorized form of bond_yield.

    Please see its documentation

    """
    return newton_wrapper(
        lambda yld: bond_price(
            settle, cpn, mat, yld, freq, comp_freq, face, redeem,
            daycount) - price,
        guess)


def bond_yield(settle=None, cpn=0, mat=1, price=100, freq=2, comp_freq=None,
               face=100, redeem=None, daycount=None, guess=10e-2):
    r"""Find the yield to maturity of one or more bonds

    Parameters
    ----------
    settle : date or None
          The settlement date. None means maturity is in years.
    cpn : float
          The coupon rate in decimal
    mat : float or date
          Maturity date or if settle is None, maturity in years
    price :
          The price of the bond
    freq : int
          Coupon frequency
    comp_freq : int
          Compounding frequency
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
    float :
          The yield to maturity of the bond. np.nan if the root finder failed.

    Examples
    --------

    >>> bond_yield(settle="2012-04-15", mat="2022-01-01", cpn=8e-2,
    ... price=94.33, freq=1)
    0.08884647275135965

    >>> bond_yield(mat=10.25, cpn=8e-2, price=93.37, freq=2)
    0.09000591604105035

    >>> bond_yield(settle="2012-04-15", mat="2022-01-01", cpn=8e-2,
    ... price=[93, 94, 95], freq=1)
    array([0.09104904, 0.08938905, 0.08775269])

    """
    return vectorize(_bond_yield_0)(
        settle=settle, cpn=cpn, mat=mat, price=price, freq=freq,
        comp_freq=comp_freq, face=face, redeem=redeem, daycount=daycount,
        guess=guess)[()]


def _import_isda_daycounters():
    r"""Import isda_daycounters. Could raise ImportError

    Returns
    -------
    daycounters : dict
         keys are the names of the daycounters, values are the classes
    default_daycounter : daycounter
         thirty360
    no_isda : bool
         Whether isda_daycounters unavailable (False)

    """
    from isda_daycounters import (actual360, actual365,
                                  actualactual, thirty360,
                                  thirtyE360, thirtyE360ISDA)
    daycounters = {x.name: x for x in (actual360, actual365,
                                       actualactual, thirty360,
                                       thirtyE360, thirtyE360ISDA)}
    default_daycounter = thirty360
    no_isda = False
    return daycounters, default_daycounter, no_isda


def _make_simple_day_counter():
    r"""Create a simple daycounter (basically ACT/365)

    Returns
    -------
    daycounters : dict
         only key is 'simple', value is SimpleDayCount
    default_daycounter : daycounter
         SimpleDayCount
    no_isda : bool
         Whether isda_daycounters available (True)

    """
    class SimpleDayCount:
        def day_count(start_date, end_date):
            return (end_date - start_date).days

        def year_fraction(start_date, end_date):
            return (end_date - start_date).days / 365.0

    daycounters = {'simple': SimpleDayCount}
    default_daycounter = SimpleDayCount
    no_isda = True
    from warnings import warn
    warn("Module isda_daycounters is not installed.\n"
         "Only 'simple' daycount (basically ACT/365) is available.\n"
         "To use other daycounts, install isda_daycounters from\n"
         "https://github.com/miradulo/isda_daycounters")
    return daycounters, default_daycounter, no_isda


try:
    daycounters, default_daycounter, no_isda = _import_isda_daycounters()
except ImportError:
    daycounters, default_daycounter, no_isda = _make_simple_day_counter()
