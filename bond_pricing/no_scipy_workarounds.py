from numpy import abs, sign, nan, vectorize


class no_scipy:
    warned = []
    msgs = dict(CubicSpline="Using linear interpolation instead",
                newton="Using root bracketing and bisection")

    @staticmethod
    def warn(fn_name):
        if fn_name not in no_scipy.warned:
            no_scipy.warned += [fn_name]
            from warnings import warn
            if fn_name in no_scipy.msgs:
                msg = no_scipy.msgs[fn_name]
            else:
                msg = ""
            no_scipy_warning = (
                "Could not import {:} from Scipy. {:}".
                format(fn_name, msg))
            warn(no_scipy_warning)


LOWER = -1 + 1e-6  # -100% (theoretical lower bound)
UPPER = 1e4  # 1,000,000% (practical upper bound)


def my_irr_0(f, lower=None, upper=None, guess=None, warn=True,
             toler=1e-6):
    if not guess:
        if lower and upper:
            guess = (lower + upper) / 2
        elif lower:
            guess = lower
        elif upper:
            guess = upper
        else:
            guess = 0
    lower = lower or LOWER
    upper = upper or UPPER
    if not(lower >= -1 and guess >= lower and guess <= upper):
        if warn:
            from warnings import warn
            warn("-1 <= lower <= guess <= upper is required")
        return None
    # we first bracket the root (f has opposite signs at L and R)
    L, R = bracket_root(f, lower, upper, guess)
    if L is None:
        print("failed to bracket root")
        return nan
    # Use bisection to find the root between L and R
    return bisection(f, L, R, toler)


my_irr = vectorize(my_irr_0)


def bracket_root(f, lower, upper, guess, nstep=100):
    def grid_search(step_pct, nstep=100):
        step = step_pct / 100
        guess_sign = sign(f(guess))
        L = None
        R = None
        for n in range(nstep):
            # search for change of sign
            # moving in steps of size step_pct
            # stop if sign change found
            if(guess + n*step <= UPPER and
               (sign(f(guess + n*step)) != guess_sign)):
                R = guess + n*step
                L = R - step
                break
            if(guess - n*step >= LOWER and
               (sign(f(guess - n*step)) != guess_sign)):
                L = guess - n*step
                R = L + step
                break
        return L, R
    for step in [1, 5, 10, 25, 100]:
        # try steps of 1%, 5%, ...
        L, R = grid_search(step)
        if L is not None:
            break
    return L, R


def bisection(f, a, b, tol):
    assert sign(f(a)) != sign(f(b))
    m = (a + b)/2

    if abs(f(m)) < tol:
        # stopping condition, report m as root
        return m
    elif sign(f(a)) == sign(f(m)):
        # case where m is an improvement on a.
        # Make recursive call with a = m
        return bisection(f, m, b, tol)
    elif sign(f(b)) == sign(f(m)):
        # case where m is an improvement on b.
        # Make recursive call with b = m
        return bisection(f, a, m, tol)
