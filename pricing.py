"""
pricing.py — Turning cash flows into a price, and a price back into a yield.

Two ways to price:
    price_from_curve(inst, curve) : each cash flow discounted with the curve's DF(t)
                                    -> the "fair value" given today's market curve
    price_from_yield(inst, ytm)   : every cash flow discounted at ONE flat rate
                                    -> the classic bond formula
                                       P = sum CF_i / (1 + y/f)^(f * t_i)

And the inverse problem:
    solve_ytm(inst, price)        : find y such that price_from_yield(y) == price.
                                    There is no closed form, so we use Newton-Raphson:
                                        y_{n+1} = y_n - g(y_n) / g'(y_n)
                                    with g(y) = P(y) - target. The derivative is
                                    analytic (it is basically the dollar duration),
                                    so convergence is quadratic: ~4-6 iterations
                                    to 1e-10 precision. Bisection is kept as a
                                    safety net for pathological inputs.

Prices are in currency units (same unit as the notional). Use .per_100 for
the market-style quote.
"""
import math
from dataclasses import dataclass
from typing import List, Optional

from curves import YieldCurve
from daycount import year_fraction
from instruments import CashFlow, Instrument


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------
@dataclass
class PriceResult:
    dirty: float                 # PV of all future cash flows (what you actually pay)
    clean: float                 # dirty - accrued (what the market quotes)
    accrued: float               # interest earned since last coupon, owed to the seller
    notional: float
    cashflows: List[CashFlow]

    @property
    def dirty_per_100(self) -> float:
        return 100.0 * self.dirty / self.notional

    @property
    def clean_per_100(self) -> float:
        return 100.0 * self.clean / self.notional


@dataclass
class YTMResult:
    ytm: float
    iterations: int
    converged: bool
    residual: float              # P(ytm) - target at the end
    method: str                  # "newton" or "bisection"
    history: List[float]         # ytm after each iteration (nice to plot)


# --------------------------------------------------------------------------
# Accrued interest
# --------------------------------------------------------------------------
def accrued_interest(inst: Instrument, cashflows: List[CashFlow]) -> float:
    """Coupon earned between the start of the current period and settlement.
    Pro-rata under the instrument's day count. Zero for zero-coupon bonds."""
    if not cashflows:
        return 0.0
    first = cashflows[0]
    if first.coupon == 0.0 or first.start_date >= inst.settlement:
        return 0.0
    full = year_fraction(first.start_date, first.pay_date, inst.day_count)
    elapsed = year_fraction(first.start_date, inst.settlement, inst.day_count)
    return first.coupon * elapsed / full if full > 0 else 0.0


# --------------------------------------------------------------------------
# Pricing off the curve
# --------------------------------------------------------------------------
def price_from_curve(inst: Instrument, curve: YieldCurve) -> PriceResult:
    cfs = inst.cashflows(curve)
    dirty = sum(cf.total * float(curve.discount_factor(cf.t)) for cf in cfs)
    acc = accrued_interest(inst, cfs)
    return PriceResult(dirty=dirty, clean=dirty - acc, accrued=acc,
                       notional=inst.notional, cashflows=cfs)


# --------------------------------------------------------------------------
# Pricing at a flat yield
# --------------------------------------------------------------------------
def _df_yield(t: float, y: float, freq: int) -> float:
    """Discount factor at yield y compounded `freq` times a year."""
    return (1.0 + y / freq) ** (-freq * t)


def pv_at_yield(cashflows: List[CashFlow], y: float, freq: int) -> float:
    return sum(cf.total * _df_yield(cf.t, y, freq) for cf in cashflows)


def dpv_dy(cashflows: List[CashFlow], y: float, freq: int) -> float:
    """Analytic derivative of the PV with respect to the yield:
       d/dy (1 + y/f)^(-f t) = -t (1 + y/f)^(-f t - 1)"""
    return sum(-cf.t * cf.total * (1.0 + y / freq) ** (-freq * cf.t - 1.0) for cf in cashflows)


def price_from_yield(inst: Instrument, ytm: float,
                     curve: Optional[YieldCurve] = None,
                     cashflows: Optional[List[CashFlow]] = None) -> PriceResult:
    """Price with every cash flow discounted at `ytm`.
    For an FRN the coupons must first be projected, so pass a curve
    (or pre-computed cashflows)."""
    cfs = cashflows if cashflows is not None else inst.cashflows(curve)
    dirty = pv_at_yield(cfs, ytm, inst.frequency)
    acc = accrued_interest(inst, cfs)
    return PriceResult(dirty=dirty, clean=dirty - acc, accrued=acc,
                       notional=inst.notional, cashflows=cfs)


# --------------------------------------------------------------------------
# Yield to maturity — Newton-Raphson
# --------------------------------------------------------------------------
def _initial_guess(inst: Instrument, cashflows: List[CashFlow], dirty_price: float) -> float:
    """Textbook approximation:  (annual coupon + (notional - price)/T) / ((notional + price)/2)"""
    T = max(cashflows[-1].t, 1e-6)
    annual_coupon = sum(cf.coupon for cf in cashflows) / T
    guess = (annual_coupon + (inst.notional - dirty_price) / T) / ((inst.notional + dirty_price) / 2.0)
    return max(guess, -0.5)


def solve_ytm(inst: Instrument, price: float,
              is_clean: bool = True,
              curve: Optional[YieldCurve] = None,
              cashflows: Optional[List[CashFlow]] = None,
              guess: Optional[float] = None,
              tol: float = 1e-10,
              max_iter: int = 50) -> YTMResult:
    """Find the yield that reproduces `price`.

    price     : market price in currency units (same unit as notional)
    is_clean  : True if `price` is a clean quote (accrued is added internally)
    """
    cfs = cashflows if cashflows is not None else inst.cashflows(curve)
    if not cfs:
        raise ValueError("instrument has no future cash flows")

    f = inst.frequency
    target = price + accrued_interest(inst, cfs) if is_clean else price

    y = guess if guess is not None else _initial_guess(inst, cfs, target)
    history = [y]

    # ---- Newton-Raphson ----
    for i in range(1, max_iter + 1):
        g = pv_at_yield(cfs, y, f) - target
        if abs(g) < tol:
            return YTMResult(y, i, True, g, "newton", history)

        d = dpv_dy(cfs, y, f)
        if d == 0.0 or not math.isfinite(d):
            break                                   # flat spot: give up on Newton

        step = g / d
        y_new = y - step
        if y_new <= -f + 1e-9 or not math.isfinite(y_new):
            break                                   # left the domain (1 + y/f) > 0

        history.append(y_new)
        if abs(y_new - y) < 1e-14:
            y = y_new
            return YTMResult(y, i, True, pv_at_yield(cfs, y, f) - target, "newton", history)
        y = y_new

    # ---- Bisection fallback (PV is strictly decreasing in y) ----
    lo, hi = -0.99 * f, 5.0
    g_lo, g_hi = pv_at_yield(cfs, lo, f) - target, pv_at_yield(cfs, hi, f) - target
    if g_lo * g_hi > 0:
        return YTMResult(float("nan"), max_iter, False, float("nan"), "bisection", history)
    for i in range(200):
        mid = 0.5 * (lo + hi)
        g_mid = pv_at_yield(cfs, mid, f) - target
        history.append(mid)
        if abs(g_mid) < tol or (hi - lo) < 1e-14:
            return YTMResult(mid, i + 1, True, g_mid, "bisection", history)
        if g_lo * g_mid < 0:
            hi, g_hi = mid, g_mid
        else:
            lo, g_lo = mid, g_mid
    return YTMResult(mid, 200, False, g_mid, "bisection", history)
