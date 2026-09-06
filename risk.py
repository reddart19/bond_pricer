"""
risk.py — Duration, convexity and key-rate sensitivities.

Two families of measures, and it matters which one you use:

ANALYTIC (yield-based) — closed-form derivatives of the bond formula
    P(y) = sum CF_i / (1 + y/f)^(f t_i)
    macaulay_duration  : PV-weighted average time of the cash flows (in years)
    modified_duration  : -(1/P) dP/dy  = macaulay / (1 + y/f)
    convexity          :  (1/P) d2P/dy2
    dv01               : price change for a 1 bp move  = modified_duration * P * 0.0001
  These take the cash flows as FIXED. Perfect for fixed-rate and zero-coupon
  bonds. For an FRN they answer the wrong question (see below).

EFFECTIVE (curve-based) — bump the whole curve, fully reprice, measure
    effective_duration  : (P_down - P_up) / (2 P dy)
    effective_convexity : (P_up + P_down - 2 P) / (P dy^2)
    key_rate_durations  : same bump, but one curve tenor at a time
  Because the instrument is REPRICED, an FRN's coupons are re-projected off
  the bumped curve: its effective duration collapses to the time until its
  next reset, which is the economically correct answer. Key-rate durations
  are what let us measure NON-parallel shocks (Phase 2 scenarios).

Note: the curve stores continuously-compounded rates, so for a plain bond
effective duration ~ Macaulay duration (d/dr of e^{-rt} is -t), while
modified duration is smaller by the factor (1 + y/f). Both are "right";
they are sensitivities to different rate definitions.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from curves import YieldCurve
from instruments import CashFlow, Instrument
from pricing import price_from_curve, pv_at_yield, solve_ytm


# --------------------------------------------------------------------------
# Analytic measures (fixed cash flows, flat yield)
# --------------------------------------------------------------------------
def macaulay_duration(cashflows: List[CashFlow], ytm: float, freq: int) -> float:
    pv = [cf.total * (1.0 + ytm / freq) ** (-freq * cf.t) for cf in cashflows]
    price = sum(pv)
    return sum(cf.t * p for cf, p in zip(cashflows, pv)) / price


def modified_duration(cashflows: List[CashFlow], ytm: float, freq: int) -> float:
    return macaulay_duration(cashflows, ytm, freq) / (1.0 + ytm / freq)


def convexity(cashflows: List[CashFlow], ytm: float, freq: int) -> float:
    """(1/P) d2P/dy2  with  d2/dy2 (1+y/f)^(-ft) = t (t + 1/f) (1+y/f)^(-ft-2)"""
    price = pv_at_yield(cashflows, ytm, freq)
    second = sum(cf.t * (cf.t + 1.0 / freq) * cf.total * (1.0 + ytm / freq) ** (-freq * cf.t - 2.0)
                 for cf in cashflows)
    return second / price


def dv01(cashflows: List[CashFlow], ytm: float, freq: int) -> float:
    """Currency P&L for a 1 bp DROP in yield (positive number for a long position)."""
    price = pv_at_yield(cashflows, ytm, freq)
    return modified_duration(cashflows, ytm, freq) * price * 1e-4


def taylor_price_change(price: float, mod_duration: float, convex: float, dy: float) -> float:
    """Second-order approximation of the price change for a yield move dy (decimal):
       dP ~ P * (-D_mod * dy + 0.5 * C * dy^2)"""
    return price * (-mod_duration * dy + 0.5 * convex * dy ** 2)


# --------------------------------------------------------------------------
# Effective measures (full repricing off a bumped curve)
# --------------------------------------------------------------------------
def effective_duration(inst: Instrument, curve: YieldCurve, bump_bp: float = 10.0) -> float:
    dy = bump_bp / 1e4
    p0 = price_from_curve(inst, curve).dirty
    up = price_from_curve(inst, curve.parallel_shift(+bump_bp)).dirty
    dn = price_from_curve(inst, curve.parallel_shift(-bump_bp)).dirty
    return (dn - up) / (2.0 * p0 * dy)


def effective_convexity(inst: Instrument, curve: YieldCurve, bump_bp: float = 10.0) -> float:
    dy = bump_bp / 1e4
    p0 = price_from_curve(inst, curve).dirty
    up = price_from_curve(inst, curve.parallel_shift(+bump_bp)).dirty
    dn = price_from_curve(inst, curve.parallel_shift(-bump_bp)).dirty
    return (up + dn - 2.0 * p0) / (p0 * dy ** 2)


def key_rate_durations(inst: Instrument, curve: YieldCurve,
                       bump_bp: float = 10.0) -> Dict[float, float]:
    """Sensitivity to each tenor of the curve separately.
    The curve interpolates linearly between knots, so bumping ONE knot creates
    a 'tent' shock centred on that tenor. The tents sum to a parallel shift,
    hence  sum(KRDs) == effective_duration.  Returned as {tenor: duration}."""
    dy = bump_bp / 1e4
    p0 = price_from_curve(inst, curve).dirty
    krd = {}
    for k in curve.tenors:
        up_curve = curve.shifted(lambda t, k=k: np.where(np.isclose(t, k), +dy, 0.0))
        dn_curve = curve.shifted(lambda t, k=k: np.where(np.isclose(t, k), -dy, 0.0))
        up = price_from_curve(inst, up_curve).dirty
        dn = price_from_curve(inst, dn_curve).dirty
        krd[float(k)] = (dn - up) / (2.0 * p0 * dy)
    return krd


# --------------------------------------------------------------------------
# One-call report
# --------------------------------------------------------------------------
@dataclass
class RiskMetrics:
    price: float                      # dirty, currency
    clean: float
    ytm: float
    macaulay: float
    modified: float
    convexity: float
    dv01: float
    effective_duration: float
    effective_convexity: float
    key_rate_durations: Dict[float, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "price": self.price, "clean": self.clean, "ytm": self.ytm,
            "macaulay": self.macaulay, "modified": self.modified,
            "convexity": self.convexity, "dv01": self.dv01,
            "eff_duration": self.effective_duration,
            "eff_convexity": self.effective_convexity,
        }
        d.update({f"krd_{k:g}y": v for k, v in self.key_rate_durations.items()})
        return d


def risk_metrics(inst: Instrument, curve: YieldCurve, bump_bp: float = 10.0,
                 with_krd: bool = True) -> RiskMetrics:
    """Price the instrument off the curve, back out its YTM, and compute every measure."""
    pr = price_from_curve(inst, curve)
    cfs = pr.cashflows
    f = inst.frequency
    y = solve_ytm(inst, pr.dirty, is_clean=False, cashflows=cfs).ytm
    return RiskMetrics(
        price=pr.dirty,
        clean=pr.clean,
        ytm=y,
        macaulay=macaulay_duration(cfs, y, f),
        modified=modified_duration(cfs, y, f),
        convexity=convexity(cfs, y, f),
        dv01=dv01(cfs, y, f),
        effective_duration=effective_duration(inst, curve, bump_bp),
        effective_convexity=effective_convexity(inst, curve, bump_bp),
        key_rate_durations=key_rate_durations(inst, curve, bump_bp) if with_krd else {},
    )
