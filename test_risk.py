"""
test_risk.py — Phase 2 checks. Run with:  python test_risk.py

Analytic measures are checked against closed-form values (zero-coupon bond)
and against finite differences; effective measures against Macaulay; key-rate
durations against their sum rule; and the portfolio against hand aggregation.
"""
from datetime import date

import numpy as np
import pandas as pd

from curves import YieldCurve
from instruments import FixedRateBond, FloatingRateNote, ZeroCouponBond
from portfolio import Portfolio, Position
from pricing import price_from_curve, price_from_yield, pv_at_yield
from risk import (convexity, dv01, effective_convexity, effective_duration,
                  key_rate_durations, macaulay_duration, modified_duration,
                  risk_metrics, taylor_price_change)
from scenarios import (bear_steepener, butterfly, flattener, parallel, standard_set,
                       steepener)

pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

SETTLE = date(2025, 1, 1)
PASSED = 0


def check(label, got, expected, tol=1e-6):
    global PASSED
    ok = abs(got - expected) < tol
    print(f"  {'✓' if ok else '✗'} {label:<52} got {got:>12.6f}   expected {expected:>12.6f}")
    assert ok, label
    PASSED += 1


curve = YieldCurve.from_annual_rates([0.5, 1, 2, 5, 10, 30],
                                     [0.030, 0.032, 0.034, 0.037, 0.040, 0.042], name="EUR")

print("\n=== 1. Zero-coupon bond: closed forms ===")
zero = ZeroCouponBond(notional=100, settlement=SETTLE, maturity=date(2035, 1, 1), frequency=1)
cfs = zero.cashflows()
check("Macaulay of a 10y zero = 10", macaulay_duration(cfs, 0.05, 1), 10.0)
check("Modified = 10 / 1.05", modified_duration(cfs, 0.05, 1), 10 / 1.05)
check("Convexity = T(T+1)/(1+y)^2", convexity(cfs, 0.05, 1), 110 / 1.05 ** 2)
check("Effective duration on a cc curve = Macaulay", effective_duration(zero, curve, 1), 10.0, 1e-3)

print("\n=== 2. Fixed-rate bond: analytic vs finite differences ===")
bond = FixedRateBond(notional=100, settlement=SETTLE, maturity=date(2035, 1, 1),
                     frequency=2, coupon_rate=0.05, name="5% Jan-2035")
cfs, y, h = bond.cashflows(), 0.06, 1e-5
P = pv_at_yield(cfs, y, 2)
Pu, Pd = pv_at_yield(cfs, y + h, 2), pv_at_yield(cfs, y - h, 2)
check("modified duration vs -(dP/dy)/P", modified_duration(cfs, y, 2), -(Pu - Pd) / (2 * h * P), 1e-6)
check("convexity vs (d2P/dy2)/P", convexity(cfs, y, 2), (Pu + Pd - 2 * P) / (h * h * P), 1e-3)
check("DV01 = modified * P * 1e-4", dv01(cfs, y, 2), modified_duration(cfs, y, 2) * P * 1e-4)
dy = 0.01
actual = price_from_yield(bond, y + dy).dirty - P
approx = taylor_price_change(P, modified_duration(cfs, y, 2), convexity(cfs, y, 2), dy)
print(f"  · +100bp: actual dP = {actual:+.4f}   Taylor (dur+conv) = {approx:+.4f}   "
      f"duration only = {-modified_duration(cfs, y, 2) * P * dy:+.4f}")
assert abs(actual - approx) < 0.02   # residual = 3rd-order term

print("\n=== 3. Key-rate durations sum to the effective duration ===")
krd = key_rate_durations(bond, curve)
check("sum(KRD) == effective duration", sum(krd.values()), effective_duration(bond, curve), 1e-4)
print("  · KRD profile: " + "  ".join(f"{k:g}y={v:.3f}" for k, v in krd.items()))

print("\n=== 4. FRN: why effective duration matters ===")
frn = FloatingRateNote(notional=100, settlement=SETTLE, maturity=date(2030, 1, 1),
                       frequency=4, spread=0.0050, name="FRN 3M+50")
m = risk_metrics(frn, curve)
print(f"  · Macaulay on PROJECTED coupons : {m.macaulay:.3f} y   (looks like a 5y bond)")
print(f"  · Effective (curve repriced)    : {m.effective_duration:.3f} y   "
      f"(coupons reset -> only the spread is exposed)")
assert m.effective_duration < 1.0 < m.macaulay

print("\n=== 5. Scenarios ===")
t = curve.tenors
s = steepener(50).describe(t)
check("steepener: 2y shift = -25bp", s[2.0], -25.0)
check("steepener: 10y shift = +25bp", s[10.0], +25.0)
check("steepener: 30y flat beyond anchor", s[30.0], +25.0)
b = butterfly(25).describe(t)
check("butterfly: belly (10y) = -25bp", b[10.0], -25.0)
check("butterfly: wing (30y) = +25bp", b[30.0], +25.0)
check("flattener = -steepener", flattener(50).describe(t)[10.0], -25.0)
check("bear steepener 10y = +50 +25", bear_steepener(50, 50).describe(t)[10.0], 75.0)
check("parallel scenario == curve.parallel_shift",
      float(parallel(100).apply(curve).zero_rate(7)), float(curve.parallel_shift(100).zero_rate(7)))

print("\n=== 6. Portfolio aggregation ===")
short_bond = FixedRateBond(notional=100, settlement=SETTLE, maturity=date(2027, 1, 1),
                           frequency=1, coupon_rate=0.03, name="3% Jan-2027")
pf = Portfolio([Position(bond, 1000), Position(zero, 500),
                Position(frn, 800), Position(short_bond, 1200)], name="Demo book")
mv_hand = sum(p.units * price_from_curve(p.instrument, curve).dirty for p in pf.positions)
check("market value = sum(units * dirty)", pf.market_value(curve), mv_hand)
w = pf.weights(curve)
d_hand = sum(wi * effective_duration(p.instrument, curve) for wi, p in zip(w, pf.positions))
check("portfolio duration = MV-weighted average", pf.duration(curve), d_hand)
dv01_hand = sum(p.units * risk_metrics(p.instrument, curve, with_krd=False).dv01 for p in pf.positions)
check("portfolio DV01 additive", pf.dv01(curve), dv01_hand)
check("portfolio KRDs sum to portfolio duration",
      sum(pf.key_rate_durations(curve).values()), pf.duration(curve), 1e-4)
pnl = pf.scenario_pnl(curve, parallel(100))
check("parallel +100bp: full repricing ~ Taylor (within 0.5%)",
      pnl["P&L"] / pf.market_value(curve),
      taylor_price_change(1.0, pf.duration(curve), pf.convexity(curve), 0.01), 5e-3)

print(f"\nAll {PASSED} checks passed.\n")

# ------------------------------------------------------------------
print("Risk table")
print(pf.risk_table(curve).to_string(index=False))
print("\nKey-rate durations")
print(pf.key_rate_table(curve).to_string())
print("\nScenario P&L (full repricing vs duration/convexity approximation)")
st = pf.scenario_table(curve, standard_set())
print(st[["Scenario", "Avg shift bp", "P&L", "P&L %", "Taylor P&L"]].to_string(index=False))
