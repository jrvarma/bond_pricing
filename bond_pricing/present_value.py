"""Present value module in bond_pricing

This module includes `npv`, `irr` and `duration` for
arbitrary cash flows. The functions for annuities include
`annuity_pv`, `annuity_pv`, `annuity_rate`, `annuity_instalment`,
`annuity_instalment_breakup`, `annuity_periods`. The helper functions
`pvaf` and `fvaf` are intended for use by the annuity functions, but
can be used directly if desired.

"""
import numpy as np
from numpy import array, log, exp, where, vectorize
from bond_pricing.utils import newton_wrapper, dict_to_dataframe


def pvaf(r, n):
    """ Compute Present Value of Annuity Factor

    Parameters
    ----------
    r : float or sequence of floats
        per period interest rate in decimal
    n : float or sequence of floats
        number of periods

    Returns
    -------
    float or array of floats
       The present value of annuity factor

    Examples
    --------
    >>> pvaf(r=0.1, n=10)
    6.144567105704685

    >>> pvaf(10e-2, [5, 10])
    array([3.79078677, 6.14456711])

    """
    r, n = array(r), array(n)
    # We use numpy.where to handle r == 0, but numpy.where evaluates
    # both expressions and so 0/0 will still happen. Suppress the error.
    old_settings = np.seterr(invalid='ignore')
    result = where(r == 0, n, np.divide(1 - (1+r)**-n, r))
    np.seterr(**old_settings)
    return result[()]


def fvaf(r, n):
    """ Compute Future Value of Annuity Factor

    Parameters
    ----------
    r : float or sequence of floats
        per period interest rate in decimal
    n : float or sequence of floats
        number of periods

    Returns
    -------
    float or array of floats
       The future value of annuity factor

    Examples
    --------
    >>> fvaf(r=0.1, n=10)
    15.937424601000023
    >>> fvaf(r=[0, 0.1], n=10)
    array([10.       , 15.9374246])

    """
    r, n = array(r), array(n)
    old_settings = np.seterr(invalid='ignore')
    result = where(r == 0, n, np.divide((1+r)**n - 1, r))
    np.seterr(**old_settings)
    return result[()]


def npv(cf, rate, cf_freq=1, comp_freq=1, cf_t=None,
        immediate_start=False):
    r"""NPV of a sequence of cash flows

    Parameters
    ----------
    cf : float or sequence of floats
         array of cash flows
    rate : float or sequence of floats
         discount rate
    cf_freq : float or sequence of floats, optional
         cash flow frequency (for example, 2 for semi-annual)
    comp_freq : float or sequence of floats, optional
         compounding frequency (for example, 2 for semi-annual)
    cf_t : float or sequence of floats or None, optional
         The timing of cash flows.
         If None, equally spaced cash flows are assumed
    immediate_start : bool or sequence of bool, optional
         If True, cash flows start immediately
         Else, the first cash flow is at the end of the first period.

    Returns
    -------
    float or array of floats
       The net present value of the cash flows

    Examples
    --------
    >>> npv(cf=[-100, 150, -50, 75], rate=5e-2)
    59.327132213429586

    >>> npv(cf=[-100, 150, -50, 75], rate=5e-2, comp_freq=[1, 2])
    array([59.32713221, 59.15230661])

    >>> npv(cf=[-100, 150, -50, 75], rate=5e-2,
    ...     immediate_start=[False, True])
    array([59.32713221, 62.29348882])

    >>> npv(cf=[-100, 150, -50, 75], cf_t=[0, 2, 5, 7], rate=[5e-2, 8e-2])
    array([50.17921321, 38.33344284])

    """

    def one_npv(rate, cf_freq, comp_freq, immediate_start):
        if cf_t is None:
            start = 0 if immediate_start else 1/cf_freq
            stop = start + len(cf) / cf_freq
            cf_ta = np.arange(start=start, step=1/cf_freq, stop=stop)
        else:
            cf_ta = array(cf_t)
        cc_rate = equiv_rate(rate, from_freq=comp_freq, to_freq=np.inf)
        df = exp(-cc_rate * cf_ta)
        return np.dot(cf, df)

    cf = array(cf)
    return vectorize(one_npv)(
        rate=rate, cf_freq=cf_freq, comp_freq=comp_freq,
        immediate_start=immediate_start)[()]


def equiv_rate(rate, from_freq=1, to_freq=1):
    r"""Convert interest rate from one compounding frequency to another

    Parameters
    ----------
    rate : float or sequence of floats
           discount rate in decimal
    from_freq : float or sequence of floats
                compounding frequency of input rate
    to_freq : float or sequence of floats
              compounding frequency of output rate

    Returns
    -------
    float or array of floats
       The discount rate for the desired compounding frequency

    Examples
    --------
    >>> equiv_rate(
    ...    rate=10e-2, from_freq=1, to_freq=[1, 2, 12, 365, np.inf])
    array([0.1       , 0.0976177 , 0.09568969, 0.09532262, 0.09531018])

    >>> equiv_rate(
    ...    rate=10e-2, from_freq=[1, 2, 12, 365, np.inf], to_freq=1)
    array([0.1       , 0.1025    , 0.10471307, 0.10515578, 0.10517092])

    """
    rate, from_freq, to_freq = array(rate), array(from_freq), array(to_freq)
    # the use of where prevents nan from being returned for 0/0
    # but since where evaluates both expressions the error still occurs
    # we run np.seterr and np.divide to catch 0/0 errors
    old_settings = np.seterr(invalid='ignore')
    cc_rate = where(from_freq == np.inf, rate,
                    log(1 + np.divide(rate, from_freq)) * from_freq)
    res = where(from_freq == to_freq,
                rate,
                where(to_freq == np.inf,
                      cc_rate,
                      (exp(np.divide(cc_rate, to_freq)) - 1) * to_freq))[()]
    np.seterr(**old_settings)
    return res


def duration(cf, rate, cf_freq=1, comp_freq=1, cf_t=None,
             immediate_start=False, modified=False):
    r"""Duration of arbitrary sequence of cash flows

    Parameters
    ----------
    cf : sequence of floats
         array of cash flows
    rate : float or sequence of floats
         discount rate
    cf_freq : float or sequence of floats, optional
         cash flow frequency (for example, 2 for semi-annual)
    comp_freq : float or sequence of floats, optional
         compounding frequency (for example, 2 for semi-annual)
    cf_t : float or sequence of floats or None, optional
         The timing of cash flows.
         If None, equally spaced cash flows are assumed
    immediate_start : bool or sequence of bool, optional
         If True, cash flows start immediately
         Else, the first cash flow is at the end of the first period.
    modified : bool or sequence of bool, optional
         If True, modified duration is returned

    Returns
    -------
    float or array of floats
       The duration of the cash flows

    Examples
    --------
    >>> duration(cf=[100, 50, 75, 25], rate=10e-2)
    1.9980073065426769

    >>> duration(cf=[100, 50, 75, 25], rate=10e-2,
    ...          immediate_start=[True, False])
    array([0.99800731, 1.99800731])

    """

    def one_duration(rate, cf_freq, comp_freq, immediate_start):
        if cf_t is None:
            start = 0 if immediate_start else 1/cf_freq
            stop = start + len(cf) / cf_freq
            cf_ta = np.arange(start=start, step=1/cf_freq, stop=stop)
        else:
            cf_ta = cf_t
        cc_rate = equiv_rate(rate, from_freq=comp_freq, to_freq=np.inf)
        df = exp(-cc_rate * cf_ta)
        return np.dot(cf*df, cf_ta) / np.dot(cf, df)

    D = vectorize(one_duration)(
        rate=rate, cf_freq=cf_freq, comp_freq=comp_freq,
        immediate_start=immediate_start)
    D /= where(modified, 1 + rate/comp_freq, 1)
    return D[()]


def irr(cf, cf_freq=1, comp_freq=1, cf_t=None, r_guess=10e-2):
    r"""IRR of a sequence of cash flows

    Multiple IRRs can be found by giving multiple values of r_guess
    as shown in one of the examples below.

    Parameters
    ----------
    cf : float or sequence of floats
         array of cash flows
    cf_freq : float or sequence of floats, optional
         cash flow frequency (for example, 2 for semi-annual)
    comp_freq : float or sequence of floats, optional
         compounding frequency (for example, 2 for semi-annual)
    cf_t : float or sequence of floats or None, optional
         The timing of cash flows.
         If None, equally spaced cash flows are assumed
    immediate_start : bool or sequence of bool, optional
         If True, cash flows start immediately
         Else, the first cash flow is at the end of the first period.
    r_guess : float or sequence of floats, optional
         Starting value (guess) for root finder

    Returns
    -------
    float or array of floats
       The internal rate of return (IRR) of the cash flows

    Examples
    --------
    >>> irr(cf=[-100, 150, -50, 75])
    0.4999999999999994

    >>> irr(cf=[-100, 150, -50, 75], cf_freq=1, comp_freq=2)
    0.4494897427831782

    >>> irr(cf=[-100, 150, -50, 75], cf_t=[0, 2, 5, 7])
    0.2247448713915599

    >>> irr(cf=(-100, 230, -132), r_guess=[0.13, 0.18])
    array([0.1, 0.2])

    """
    if np.sign(max(cf)) == np.sign(min(cf)):
        return(np.nan)

    def one_irr(cf_freq, comp_freq, r_guess):

        def f(r):
            return npv(cf=cf, rate=r, cf_freq=cf_freq,
                       comp_freq=comp_freq, cf_t=cf_t)
        return newton_wrapper(f, r_guess)

    return vectorize(one_irr)(
        cf_freq=cf_freq, comp_freq=comp_freq, r_guess=r_guess)[()]


def annuity_pv(rate, n_periods=np.inf, instalment=1, terminal_payment=0,
               immediate_start=False, cf_freq=1, comp_freq=1):
    r"""Present value of annuity

    Parameters
    ----------
    rate : float or sequence of floats
            per period discount rate in decimal
    n_periods : float or sequence of floats
            number of periods of annuity
    instalment : float or sequence of floats
            instalment per period
    terminal_payment : float or sequence of floats
            baloon payment at the end of the annuity
    immediate_start : bool or sequence of bool
         If True, cash flows start immediately
         Else, the first cash flow is at the end of the first period.
    cf_freq : float or sequence of floats
         cash flow frequency (for example, 2 for semi-annual)
    comp_freq : float or sequence of floats
         compounding frequency (for example, 2 for semi-annual)

    Returns
    -------
    float or array of floats
       The present value of the annuity

    Examples
    --------
    >>> annuity_pv(rate=10e-2, n_periods=15, instalment=500)
    3803.039753154183
    >>> annuity_pv(rate=10e-2, n_periods=[10, 15], instalment=500)
    array([3072.28355285, 3803.03975315])

    """
    rate, n_periods, instalment, terminal_payment, immediate_start = (
        array(rate), array(n_periods), array(instalment),
        array(terminal_payment), array(immediate_start))
    cf_freq, comp_freq = array(cf_freq), array(comp_freq)
    r = equiv_rate(rate, comp_freq, cf_freq) / cf_freq
    pv = pvaf(r, n_periods) * instalment + (
        terminal_payment / (1 + r)**n_periods)
    pv *= where(immediate_start, 1 + r, 1)
    return pv[()]


def annuity_fv(rate, n_periods=np.inf, instalment=1, terminal_payment=0,
               immediate_start=False, cf_freq=1, comp_freq=1):
    r"""Future value of annuity

    Parameters
    ----------
    rate : float or sequence of floats
            per period discount rate in decimal
    n_periods : float or sequence of floats
            number of periods of annuity
    instalment : float or sequence of floats
            instalment per period
    terminal_payment : float or sequence of floats
            baloon payment at the end of the annuity
    immediate_start : bool or sequence of bool
         If True, cash flows start immediately
         Else, the first cash flow is at the end of the first period.
    cf_freq : float or sequence of floats
         cash flow frequency (for example, 2 for semi-annual)
    comp_freq : float or sequence of floats
         compounding frequency (for example, 2 for semi-annual)

    Returns
    -------
    float or array of floats
       The future value of the annuity

    Examples
    --------
    >>> annuity_fv(rate=10e-2, n_periods=15, instalment=500)
    15886.240847078281
    >>> annuity_fv(rate=10e-2, n_periods=[10, 15], instalment=500)
    array([ 7968.7123005 , 15886.24084708])

    """
    rate, n_periods, instalment, terminal_payment, immediate_start = (
        array(rate), array(n_periods), array(instalment),
        array(terminal_payment), array(immediate_start))
    cf_freq, comp_freq = array(cf_freq), array(comp_freq)
    r = equiv_rate(rate, comp_freq, cf_freq)/cf_freq
    tv = fvaf(r, n_periods) * instalment + terminal_payment
    tv *= where(immediate_start, 1 + r, 1)
    return tv[()]


def annuity_instalment(rate, n_periods=np.inf, pv=None, fv=None,
                       terminal_payment=0, immediate_start=False,
                       cf_freq=1, comp_freq=1):
    r"""Periodic instalment to get desired PV or FV

    Parameters
    ----------
    rate : float or sequence of floats
            per period discount rate in decimal
    n_periods : float or sequence of floats
            number of periods of annuity
    pv : float or sequence of floats
         desired present value of annuity
    fv : float or sequence of floats
         desired future value of annuity
         If pv and fv are given, discounted value of fv is added to pv
    terminal_payment : float or sequence of floats
            baloon payment at the end of the annuity
    immediate_start : bool or sequence of bool
         If True, cash flows start immediately
         Else, the first cash flow is at the end of the first period.
    cf_freq : float or sequence of floats
         cash flow frequency (for example, 2 for semi-annual)
    comp_freq : float or sequence of floats
         compounding frequency (for example, 2 for semi-annual)

    Returns
    -------
    float or array of floats
       The instalment


    Examples
    --------
    >>> annuity_instalment(rate=10e-2, n_periods=15, pv=3803.04)
    500.0000324537518

    >>> annuity_instalment(rate=10e-2, n_periods=[10, 15], pv=3803.04)
    array([618.92724655, 500.00003245])

    """
    if fv is None:
        pv = pv or 1
        fv = 0
    else:
        pv = pv or 0
    fv, pv, n_periods = array(fv), array(pv), array(n_periods)
    r = equiv_rate(rate, comp_freq, cf_freq)/cf_freq
    reqd_annuity_pv = pv + (fv - terminal_payment) / (1 + r)**n_periods
    return (reqd_annuity_pv / pvaf(r, n_periods))[()]


def annuity_periods(rate, instalment=1, pv=None, fv=None,
                    terminal_payment=0, immediate_start=False,
                    cf_freq=1, comp_freq=1, round2int_digits=6):
    r"""Number of periods of annuity to get desired PV or FV

    Parameters
    ----------
    rate : float or sequence of floats
            per period discount rate in decimal
    instalment : float or sequence of floats
            instalment per period
    pv : float or sequence of floats
         desired present value of annuity
    fv : float or sequence of floats
         desired future value of annuity
         If pv and fv are given, discounted value of fv is added to pv
    terminal_payment : float or sequence of floats
            baloon payment at the end of the annuity
    immediate_start : bool or sequence of bool
         If True, cash flows start immediately
         Else, the first cash flow is at the end of the first period.
    cf_freq : float or sequence of floats
         cash flow frequency (for example, 2 for semi-annual)
    comp_freq : float or sequence of floats
         compounding frequency (for example, 2 for semi-annual)
    round2int_digits: float or sequence of floats
         answer is rounded to integer if round2int_digits after the
         decimal point are zero

    Returns
    -------
    float or array of floats
       The number of periods

    Examples
    --------
    >>> annuity_periods(rate=10e-2, instalment=500, pv=3803.04)
    15.000002163748604

    >>> annuity_periods(rate=10e-2, instalment=500, pv=3803.04,
    ...      round2int_digits=4)
    15.0

    >>> annuity_periods(rate=[0, 10e-2], instalment=500, pv=3803.04)
    array([ 7.60608   , 15.00000216])

"""
    if fv is None:
        pv = pv or 1
        fv = 0
    else:
        pv = pv or 0
    fv, pv, instalment, terminal_payment = (
        array(fv), array(pv), array(instalment), array(terminal_payment))
    # if immediate_start
    # makes it a deferred annuity (with one less period)
    pv -= where(immediate_start, instalment, 0)
    # pv + fv * df
    #   = terminal_payment * df + instalment * (1 - df) / r
    # pv + fv * df
    #   = terminal_payment * df + instalment / r  - instalment * df / r
    # fv * df - terminal_payment * df + instalment * df / r
    #   =  instalment / r  - pv
    # df * (fv - terminal_payment  + instalment / r)
    #   =  instalment / r  - pv
    r = equiv_rate(rate, comp_freq, cf_freq) / cf_freq
    # We use numpy.where to handle r == 0 or df == 0
    # but numpy.where evaluates both expressions and so
    # we must suppress the error.
    old_settings = np.seterr(divide='ignore', invalid='ignore')
    perpetuity = instalment / r
    df = np.divide(perpetuity - pv, fv - terminal_payment + perpetuity)
    n = where(r == 0,
              (pv + fv) / instalment,
              where((df < 0) | (df > 1),
                    np.nan,
                    where(df == 0,
                          np.inf,
                          -log(df) / log(1 + r))))
    np.seterr(**old_settings)
    # if immediate_start
    # add back the one period that we removed
    n += where(immediate_start, 1, 0)
    # if the result is close to an integer, then round to the integer
    # the tolerance for this is given by round2int_digits
    return where(np.abs(n - n.round(0)) < 10**-round2int_digits,
                 n.round(0).astype(int), n)[()]


def annuity_rate(n_periods=np.inf, instalment=1, pv=None, fv=None,
                 terminal_payment=0, immediate_start=False,
                 cf_freq=1, comp_freq=1, r_guess=0):
    r"""Discount rate to get desired PV or FV of annuity

    Parameters
    ----------
    n_periods : float or sequence of floats
            number of periods of annuity
    instalment : float or sequence of floats
            instalment per period
    pv : float or sequence of floats
         desired present value of annuity
    fv : float or sequence of floats
         desired future value of annuity
         If pv and fv are given, discounted value of fv is added to pv
    terminal_payment : float or sequence of floats
            baloon payment at the end of the annuity
    immediate_start : bool or sequence of bool
         If True, cash flows start immediately
         Else, the first cash flow is at the end of the first period.
    cf_freq : float or sequence of floats
         cash flow frequency (for example, 2 for semi-annual)
    comp_freq : float or sequence of floats
         compounding frequency (for example, 2 for semi-annual)
    r_guess : float, optional
         Starting value (guess) for root finder

    Returns
    -------
    float or array of floats
       The discount rate

    Examples
    --------
    >>> annuity_rate(n_periods=15, instalment=500, pv=3803.04)
    0.09999998862890495

    >>> annuity_rate(n_periods=[9, 10, 15], instalment=100, pv=1000)
    array([-0.0205697 ,  0.        ,  0.05556497])

    """
    if fv is None:
        pv = pv or 1
        fv = 0
    else:
        pv = pv or 0

    def one_rate(n_periods, instalment, pv, terminal_payment,
                 immediate_start, cf_freq, comp_freq):
        if (n_periods == np.inf):
            return (pv/instalment)

        def f(r):
            return annuity_pv(
                r, n_periods=n_periods, instalment=instalment,
                terminal_payment=terminal_payment,
                immediate_start=immediate_start, cf_freq=cf_freq,
                comp_freq=comp_freq) - pv

        return newton_wrapper(f, r_guess)

    return vectorize(one_rate)(
        n_periods=n_periods, instalment=instalment, pv=pv,
        terminal_payment=terminal_payment,
        immediate_start=immediate_start, cf_freq=cf_freq,
        comp_freq=comp_freq)[()]


def annuity_instalment_breakup(
        rate, n_periods=np.inf, pv=None, fv=None,
        terminal_payment=0, immediate_start=False,
        cf_freq=1, comp_freq=1, period_no=1,
        return_dataframe=False):
    r"""Break up instalment into principal and interest parts

    Parameters
    ----------
    rate : float or sequence of floats
            per period discount rate in decimal
    n_periods : float or sequence of floats
            number of periods of annuity
    pv : float or sequence of floats
         desired present value of annuity
    fv : float or sequence of floats
         desired future value of annuity
         If pv and fv are given, discounted value of fv is added to pv
    terminal_payment : float or sequence of floats
            baloon payment at the end of the annuity
    immediate_start : bool or sequence of bool
         If True, cash flows start immediately
         Else, the first cash flow is at the end of the first period.
    cf_freq : float or sequence of floats
         cash flow frequency (for example, 2 for semi-annual)
    comp_freq : float or sequence of floats
         compounding frequency (for example, 2 for semi-annual)
    return_dataframe : bool
         whether to return pandas DataFrame instead of dict

    Returns
    -------
    dict
            Opening Principal

            Instalment

            Interest Part

            Principal Part

            Closing Principal

    Examples
    --------
    >>> annuity_instalment_breakup(rate=10e-2, n_periods=15, pv=3803.04,
    ...       period_no=6)  # doctest: +NORMALIZE_WHITESPACE
    {'Period No': 6,
     'Opening Principal': 3072.283752266599,
     'Instalment': 500.0000324537518,
     'Interest Part': 307.2283752266599,
     'Principal Part': 192.7716572270919,
     'Closing Principal': 2879.512095039507}

    >>> d = annuity_instalment_breakup(rate=10e-2, n_periods=15, pv=3803.04,
    ...       period_no=range(1, 4), return_dataframe=True
    ...       ); print(d.iloc[:, :4]); print(d.iloc[:, 4:])
       Period No  Opening Principal  Instalment  Interest Part
    0          1        3803.040000  500.000032     380.304000
    1          2        3683.343968  500.000032     368.334397
    2          3        3551.678332  500.000032     355.167833
       Principal Part  Closing Principal
    0      119.696032        3683.343968
    1      131.665636        3551.678332
    2      144.832199        3406.846133

    """
    fv, pv, n_periods = array(fv), array(pv), array(n_periods)
    cf_freq, comp_freq = array(cf_freq), array(comp_freq)
    period_no = array(period_no)
    assert (np.all(period_no > 0) and
            np.all(period_no > 0) and
            np.all(period_no <= n_periods) and
            np.all(period_no.astype(int) == period_no)
            ), "Invalid period_no"
    instalment = annuity_instalment(
        rate=rate, n_periods=n_periods, pv=pv,
        immediate_start=immediate_start, cf_freq=cf_freq,
        comp_freq=comp_freq)
    r = equiv_rate(rate, comp_freq, cf_freq)/cf_freq
    df = (1 + r)**(period_no-1)
    opening_principal = pv * df - annuity_fv(
        rate=rate, n_periods=period_no-1, instalment=instalment,
        immediate_start=immediate_start,
        cf_freq=cf_freq, comp_freq=comp_freq)
    result = {"Period No": period_no[()],
              "Opening Principal": opening_principal,
              "Instalment": instalment,
              "Interest Part": opening_principal * r,
              "Principal Part": instalment - opening_principal * r,
              "Closing Principal":
              opening_principal + opening_principal * r - instalment}
    if return_dataframe:
        return dict_to_dataframe(result)
    else:
        return result
