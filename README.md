# Overview

This package provides bond pricing functions as well as basic NPV/IRR functions. Bond valuation can be done using an yield to maturity or using a zero yield curve. There is a convenience function to construct a zero yield curve from a few points on the par bond or zero yield curve or from Nelson Siegel parameters.

The documentation is available at <https://bond-pricing.readthedocs.io/>

The bond valuation functions can be used in two modes:

* The first mode is similar to spreadsheet bond pricing functions. The settlement date and maturity date are given as dates and the software calculates the time to maturity and to each coupon payment date from these dates. For any `daycount` other than simple counting of days (ACT/365 in ISDA terminology), this packages relies on the `isda_daycounters` module that can be downloaded from <https://github.com/miradulo/isda_daycounters>

* Maturity can be given in years (the `settle` parameter is set to `None` and is assumed to be time 0) and there are no dates at all. This mode is particularly convenient to price par bonds or price other bonds on issue date or coupon dates. For example, finding the price of a 7 year 3.5% coupon bond if the prevailing yield is 3.65% is easier in this mode as the maturity is simply given as 7.0 instead of providing a maturity date and specifying today's date. Using this mode between coupon dates is not so easy as the user has to basically compute the day count and year fraction and provide the maturity as say 6.7 years.

* Bond Valuation
    - Bond price using YTM (`bond_price`) or using zero yield curve (`zero_curve_bond_price`)
    - Accrued interest and dirty bond prices using YTM (`bond_price_breakup`)  or using zero yield curve (`zero_curve_bond_price_breakup`)
    - Duration using YTM (`bond_duration`)
    - Yield to maturity (`bond_yield`). 

* Zero curve construction
    - bootstrap zero yields from par yields (`par_yld_to_zero`) or vice versa (`zero_to_par`)
    - compute zero rates from Nelson Siegel parameters (`nelson_siegel_zero_rate`)
    - construct zero prices from par or zero yields at selected knot points using a cubic spline or assuming a flat yield curve (`make_zero_price_fun`)

* Present Value functions
    - Net Present Value (`npv`)
    - Internal Rate of Return (`irr`) 
    - Duration (`duration`). 
  These functions allow different compounding frequencies: for example, the cash flows may be monthly while the interest rate is semi-annually compounded. The function `equiv_rate` converts between different compounding frequencies.

* Annuity functions
    - Annuity present value (`annuity_pv`)
    - Future value (`annuity_fv`)
    - Implied interest rate (`annuity_rate`)
    - Number of periods to achieve given present value or terminal value (`annuity_periods`).
    - Periodic instalment to achieve given present value or terminal value (`annuity_instalment`).
    - Breakup of instalment into principal and interest (`annuity_instalment_breakup`) 
    
  In these functions also, the cash flow frequency may be different from the compounding frequency.

