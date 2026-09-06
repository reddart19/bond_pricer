"""
test_montecarlo.py — Phase 3 checks. Run with:  python test_montecarlo.py

Checks: the vectorised pricer reproduces the Phase-1 pricer exactly; the
simulated processes have the analytic Vasicek moments and the requested
correlation; zero vol gives zero P&L; VaR/ES are ordered as they must be;
and the convexity gap between full revaluation and the delta approximation
has the right sign.
"""
import time
from datetime import date

import numpy as np
import pandas as pd

from curves import YieldCurve
from instruments import FixedRateBond, FloatingRateNote, ZeroCouponBond
from montecarlo import (Correlations, EquityPosition, InflationParams, MCResult,
                        MonteCarloEngine, PathPricer, VasicekParams)
from portfolio import Portfolio, Position
from pricing import price_from_curve

pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

SETTLE = date(2025, 1, 1)
PASSED = 0


def check(label, got, expected, tol=1e-6):
    global PASSED
    ok = abs(got - expected) < tol
    print(f"  {'✓' if ok else '✗'} {label:<54} got {got:>14.6f}   expected {expected:>14.6f}")
    assert ok, label
    PASSED += 1


curve = YieldCurve.from_annual_rates([0.5, 1, 2, 5, 10, 30],
                                     [0.030, 0.032, 0.034, 0.037, 0.040, 0.042], name="EUR")
bond = FixedRateBond(notional=100, settlement=SETTLE, maturity=date(2035, 1, 1),
                     frequency=2, coupon_rate=0.05, name="5% Jan-2035")
zero = ZeroCouponBond(notional=100, settlement=SETTLE, maturity=date(2035, 1, 1))
frn = FloatingRateNote(notional=100, settlement=SETTLE, maturity=date(2030, 1, 1),
                       frequency=4, spread=0.005, name="FRN 3M+50")
frn_fix = FloatingRateNote(notional=100, settlement=date(2025, 2, 10), maturity=date(2030, 1, 1),
                           frequency=4, spread=0.005, current_fixing=0.031, name="FRN fixed")
short = FixedRateBond(notional=100, settlement=SETTLE, maturity=date(2027, 1, 1),
                      frequency=1, coupon_rate=0.03, name="3% Jan-2027")

print("\n=== 1. Vectorised pricer == Phase-1 pricer on random shifted curves ===")
rng = np.random.default_rng(0)
for inst in (bond, zero, frn, frn_fix, short):
    pricer = PathPricer(inst, curve)
    shifts = rng.normal(0, 0.01, size=(5, len(curve.tenors)))
    fast = pricer.price(shifts)
    slow = [price_from_curve(inst, curve.shifted(lambda t, s=s: s)).dirty for s in shifts]
    check(f"{inst.name}: max |fast - slow| over 5 shifts", float(np.max(np.abs(fast - slow))), 0.0, 1e-9)

pf = Portfolio([Position(bond, 1000), Position(zero, 500), Position(frn, 800),
                Position(short, 1200)], name="Demo book")

print("\n=== 2. Vasicek moments (exact transition) ===")
vp = VasicekParams(r0=0.03, kappa=0.5, theta=0.05, sigma=0.02)
eng = MonteCarloEngine(pf, curve, rates=vp, inflation=InflationParams(sigma=0.0))
H, n = 1.0, 200_000
res = eng.simulate(n_paths=n, horizon_years=H, n_steps=12, seed=1)
rH = res.r_paths[:, -1]
mean_th = vp.r0 * np.exp(-vp.kappa * H) + vp.theta * (1 - np.exp(-vp.kappa * H))
std_th = vp.sigma * np.sqrt((1 - np.exp(-2 * vp.kappa * H)) / (2 * vp.kappa))
check("E[r_1y]  (12 monthly steps, 200k paths)", float(rH.mean()), mean_th, 4 * std_th / np.sqrt(n))
check("Std[r_1y]", float(rH.std()), std_th, 4 * std_th / np.sqrt(2 * n))

print("\n=== 3. Correlation of the shocks ===")
corr = Correlations(rate_inflation=0.6, rate_equity=-0.3, inflation_equity=-0.1)
eng = MonteCarloEngine(pf, curve, rates=VasicekParams(0.03, 0.2, sigma=0.01),
                       inflation=InflationParams(sigma=0.01), correlations=corr)
res = eng.simulate(n_paths=200_000, horizon_years=1.0, n_steps=1, seed=2)
rho = np.corrcoef(res.r_paths[:, -1], res.pi_paths[:, -1])[0, 1]
check("corr(dr, dpi) == 0.6", float(rho), 0.6, 0.01)

print("\n=== 4. Sanity: zero vol -> zero P&L, seed -> reproducible ===")
eng0 = MonteCarloEngine(pf, curve, rates=VasicekParams(0.03, 0.2, theta=0.03, sigma=0.0),
                        inflation=InflationParams(sigma=0.0, theta=0.02))
res0 = eng0.simulate(n_paths=1000)
check("sigma = 0 -> max |P&L| = 0", float(np.abs(res0.pnl).max()), 0.0, 1e-9)
eng = MonteCarloEngine(pf, curve)
a, b = eng.simulate(2000, seed=7), eng.simulate(2000, seed=7)
check("same seed -> identical P&L", float(np.abs(a.pnl - b.pnl).max()), 0.0)

print("\n=== 5. The real run: 10-day VaR, 10,000 paths, bonds + equity ===")
equity = [EquityPosition("Equity index", units=500, price=100.0, mu=0.06, sigma=0.20)]
eng = MonteCarloEngine(pf, curve, equities=equity)
t0 = time.perf_counter()
res = eng.simulate(n_paths=10_000, horizon_years=10 / 252, seed=42)
elapsed = time.perf_counter() - t0
print(f"  · 10,000 paths x 10 daily steps x full repricing in {elapsed*1000:.0f} ms")
v95, v99 = res.var(res.pnl, 0.95), res.var(res.pnl, 0.99)
e95, e99 = res.es(res.pnl, 0.95), res.es(res.pnl, 0.99)
assert v99 > v95 > 0 and e95 >= v95 and e99 >= v99
print(f"  ✓ VaR99 ({v99:,.0f}) > VaR95 ({v95:,.0f}) > 0,  ES95 >= VaR95,  ES99 >= VaR99")
PASSED += 1
v_full = res.var(res.pnl_bonds, 0.99)
v_delta = res.var(res.pnl_delta_bonds, 0.99)
assert v_full < v_delta
print(f"  ✓ convexity: full-revaluation VaR99 on bonds ({v_full:,.0f}) < delta VaR99 ({v_delta:,.0f})")
PASSED += 1
assert abs(v_full - v_delta) / v_delta < 0.05
print(f"  ✓ ...but within 5% for a 10-day horizon ({100*(v_delta-v_full)/v_delta:.2f}% gap)")
PASSED += 1
check("bonds P&L + equity P&L == total", float(np.abs(res.pnl_bonds + res.pnl_equity - res.pnl).max()), 0.0)

print(f"\nAll {PASSED} checks passed.\n")

# ------------------------------------------------------------------
print("Market value today: bonds {:,.0f} + equity {:,.0f} = {:,.0f}".format(
    res.mv0_bonds, res.mv0_equity, res.mv0))
print("\nVaR / ES table (losses in currency)")
print(res.summary().to_string())
print("\nStats")
for k, v in res.stats().items():
    print(f"  {k:<20} {v:>14,.4f}" if isinstance(v, float) else f"  {k:<20} {v:>14}")
print("\nSimulated curve shift at the horizon (bp)")
print(res.curve_fan().to_string())
