# 1.0.0 released 2026-09-27

- Fixed [bug # 8](https://github.com/jrvarma/bond_pricing/issues/8) in annuity instalment for immediate start annuities.
- Fixed [bug # 9](https://github.com/jrvarma/bond_pricing/issues/9) in annuity future value for immediate start annuities.
- Fixed [bug # 10](https://github.com/jrvarma/bond_pricing/issues/10) by replacing `np.arange` with `np.linspace` to avoid numeric instability.

# 0.7.3 released 2024-09-25

Fixes for `numpy 2.0` compatibility

# 0.7.2 released 2024-03-17

Updated `README`

# 0.7.1 released 2023-12-20

Corrected bug in coupon date calculation


# 0.6.4 released 2022-01-01

Allow installation without `scipy` dependency.

# 0.6.3 released 2021-03-03

Fixed bug when bond priced on coupon day

# 0.6.2 released 2020-10-06

## Added 
Bond Valuation Key Rate Shifts


# 0.5.3 released 2020-10-04

Fixed bug in zero price, duration when freq != 1

# 0.5.1 released 2020-09-19

## Added

- Duration of coupon bond using zero yields
- Static spread (Z-spread) over zero curve to match bond price
- Some functions have option to return `pandas DataFrame` instead of `Dict`

# 0.5.0 released 2020-09-05

Initial release
