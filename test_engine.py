"""
test_engine.py — Phase 1 checks. Run with:  python3 test_engine.py

Every check compares the engine against a value you can verify by hand
(textbook formula, par bond, round-trip). If all lines show a check mark,
the pricing engine is correct.
"""
import math
from datetime import date

from curves import YieldCurve
from daycount import DayCount
from instruments import FixedRateBond, FloatingRateNote, ZeroCouponBond
from pricing import price_from_curve, price_from_yield, solve_ytm

SETTLE = date(2025, 1, 1)
PASSED = 0


def check(label: str, got: float, expected: float, tol: float = 1e-6):
    global PASSED
    ok = abs(got - expected) < tol
    print(f"  {'✓' if ok else '✗'} {label:<48} got {got:>12.6f}   expected {expected:>12.6f}")
    assert ok, label
    PASSED += 1


print("\n=== 1. Fixed-rate bond ===")
bond = FixedRateBond(notional=100, settlement=SETTLE, maturity=date(2035, 1, 1),
                     frequency=2, coupon_rate=0.05, name="5% Jan-2035")

# Textbook: 5% semi-annual, 10y, yield 6%  ->  92.5613
check("10y 5% semi @ 6% (textbook 92.5613)", price_from_yield(bond, 0.06).dirty, 92.56127, 1e-4)
# Coupon = yield  ->  price = par
check("coupon == yield -> par", price_from_yield(bond, 0.05).dirty, 100.0)
# Flat 5% annual curve, annual bond 5%  ->  par
bond_annual = FixedRateBond(notional=100, settlement=SETTLE, maturity=date(2035, 1, 1),
                            frequency=1, coupon_rate=0.05)
flat5 = YieldCurve.from_annual_rates([1, 5, 10, 30], [0.05, 0.05, 0.05, 0.05])
check("5% annual bond on flat 5% curve -> par", price_from_curve(bond_annual, flat5).dirty, 100.0)

print("\n=== 2. Zero-coupon bond ===")
zero = ZeroCouponBond(notional=100, settlement=SETTLE, maturity=date(2035, 1, 1), frequency=1)
check("10y zero @ 5% = 100/1.05^10", price_from_yield(zero, 0.05).dirty, 100 / 1.05 ** 10)
check("10y zero on flat 5% curve", price_from_curve(zero, flat5).dirty, 100 / 1.05 ** 10)

print("\n=== 3. Newton-Raphson YTM ===")
res = solve_ytm(bond, 92.56127)
check("YTM of 92.5613 -> 6.00%", res.ytm, 0.06, 1e-6)
print(f"      ({res.method}, {res.iterations} iterations, residual {res.residual:.1e})")
res = solve_ytm(zero, 100 / 1.05 ** 10, is_clean=True)
check("YTM of zero -> 5.00%", res.ytm, 0.05, 1e-9)
# Round trip at an ugly yield
p = price_from_yield(bond, 0.073219).dirty
check("round trip price -> yield", solve_ytm(bond, p, is_clean=False).ytm, 0.073219, 1e-9)
# Deep discount / premium
check("premium bond YTM (price 130)", price_from_yield(bond, solve_ytm(bond, 130).ytm).clean, 130.0, 1e-6)
check("distressed bond YTM (price 30)", price_from_yield(bond, solve_ytm(bond, 30).ytm).clean, 30.0, 1e-6)

print("\n=== 4. Floating-rate note ===")
curve = YieldCurve.from_annual_rates([0.5, 1, 2, 5, 10, 30],
                                     [0.030, 0.032, 0.034, 0.037, 0.040, 0.042], name="EUR")
frn = FloatingRateNote(notional=100, settlement=SETTLE, maturity=date(2030, 1, 1),
                       frequency=4, spread=0.0, name="FRN 3M+0")
# Zero spread, priced on the SAME curve that projects its coupons  ->  exactly par
check("FRN spread 0 on reset date -> par", price_from_curve(frn, curve).dirty, 100.0, 1e-8)
frn_50 = FloatingRateNote(notional=100, settlement=SETTLE, maturity=date(2030, 1, 1),
                          frequency=4, spread=0.0050, name="FRN 3M+50")
p50 = price_from_curve(frn_50, curve).dirty
print(f"  · FRN +50bp on the same curve prices ABOVE par: {p50:.4f}  (PV of the spread)")
assert p50 > 100.0
res = solve_ytm(frn_50, p50, is_clean=False, curve=curve)
print(f"  · FRN +50bp yield-to-maturity: {res.ytm*100:.4f}%  ({res.method}, {res.iterations} it.)")

print("\n=== 5. Accrued interest / clean vs dirty ===")
mid = FixedRateBond(notional=100, settlement=date(2025, 3, 1), maturity=date(2035, 1, 1),
                    frequency=2, coupon_rate=0.05, day_count=DayCount.THIRTY_360)
r = price_from_yield(mid, 0.05)
# 60 days of a 180-day period on a 2.50 coupon -> 0.8333
check("accrued 2 months into a 2.50 coupon", r.accrued, 2.5 * 60 / 180)
check("clean = dirty - accrued", r.clean, r.dirty - r.accrued)

print("\n=== 6. Curve mechanics ===")
check("DF(0) = 1", float(curve.discount_factor(0.0)), 1.0)
check("+100bp parallel shift", float(curve.parallel_shift(100).zero_rate(5)),
      float(curve.zero_rate(5)) + 0.01)
f = curve.forward_rate(1.0, 2.0)
df1, df2 = float(curve.discount_factor(1)), float(curve.discount_factor(2))
check("forward 1y->2y consistent with DFs", 1 + f, df1 / df2)

print(f"\nAll {PASSED} checks passed.\n")

# ---- Small demo table ----
print("Demo: three instruments on the EUR curve")
print(f"{'instrument':<22}{'dirty':>10}{'clean':>10}{'accrued':>10}{'YTM':>9}")
for inst in (bond, zero, frn_50):
    pr = price_from_curve(inst, curve)
    y = solve_ytm(inst, pr.dirty, is_clean=False, curve=curve)
    print(f"{inst.name:<22}{pr.dirty_per_100:>10.4f}{pr.clean_per_100:>10.4f}"
          f"{pr.accrued:>10.4f}{y.ytm*100:>8.3f}%")
