# Fixed-income pricing & risk engine

A modular fixed-income engine with a Streamlit dashboard: prices fixed-rate
bonds, floating-rate notes and zero-coupon bonds off a yield curve, solves
exact yield-to-maturity with Newton-Raphson, computes duration / convexity /
key-rate durations, re-prices a portfolio under parallel and non-parallel
curve shocks, and runs a correlated Monte Carlo (rates + inflation + equity)
to produce 95% / 99% Value-at-Risk and Expected Shortfall.

Pure Python: `numpy`, `pandas`, `streamlit`, `altair`. No QuantLib.

## Quick start

```bash
pip install -r requirements.txt
python test_engine.py        # Phase 1: pricing        (16 checks)
python test_risk.py          # Phase 2: risk analytics (21 checks)
python test_montecarlo.py    # Phase 3: Monte Carlo    (14 checks)
python test_app.py           # Phase 4: dashboard smoke test
streamlit run app.py         # the dashboard  (or: python -m streamlit run app.py)
```

On macOS use `python3` / `pip3`.

## Layout

```
daycount.py      30/360, ACT/360, ACT/365 year fractions
curves.py        YieldCurve: discount factors, forwards, interpolation, shocks
instruments.py   FixedRateBond, ZeroCouponBond, FloatingRateNote -> list of CashFlow
pricing.py       price off a curve, price at a yield, accrued, Newton-Raphson YTM
risk.py          Macaulay / modified / convexity / DV01, effective duration, key-rate durations
scenarios.py     parallel, steepener, flattener, butterfly, custom, bear/bull combos
portfolio.py     Position / Portfolio: MV-weighted risk, KRD table, scenario P&L, ladder
montecarlo.py    Vasicek + inflation + equity factors, vectorised full revaluation, VaR / ES
app.py           Streamlit dashboard (Book, Pricing, Curve, Risk, Scenarios, Monte Carlo)
.streamlit/      theme
```

The design rule that keeps it modular: **an instrument only produces cash
flows**. Pricing, risk and Monte Carlo never branch on the instrument type;
adding a new instrument is one new class.

## Methodology in one paragraph each

**Pricing.** The curve stores continuously-compounded zero rates,
`DF(t) = exp(-r(t) t)`, linear interpolation on rates, flat extrapolation.
Fixed and zero-coupon flows are known; an FRN's future coupons are set to the
curve's simple forward rates plus the spread (the current period can use an
already-fixed rate). Dirty price is the PV of all future flows; clean price
subtracts accrued interest under the instrument's day count.

**Yield to maturity.** `P(y) = Σ CF_i / (1 + y/f)^(f t_i)` has no closed-form
inverse. Newton-Raphson with the analytic derivative (the dollar duration)
converges quadratically: 4 iterations to 1e-14 on a 10-year bond. Bisection
is the fallback if an input pushes Newton outside `(1 + y/f) > 0`.

**Risk.** Analytic measures (Macaulay, modified, convexity, DV01) are
closed-form derivatives of the yield formula and treat cash flows as fixed.
Effective measures bump the whole curve and fully re-price, which is the
right thing for an FRN (its effective duration collapses to the time to the
next reset). Key-rate durations bump one curve tenor at a time; because
interpolation is linear the tents sum to a parallel shift, so
`Σ KRD = effective duration`.

**Scenarios.** A scenario is a function `tenor -> bp`. Applying it returns a
new curve; the portfolio is fully re-priced. The duration + convexity Taylor
estimate is shown alongside — it matches on parallel moves and fails on
steepeners and butterflies, which is exactly what key-rate durations exist for.

**Monte Carlo.** Three risk factors with Cholesky-correlated Gaussian shocks
and exact Ornstein-Uhlenbeck transitions: a Vasicek short rate (shifts the
curve by `B(τ)/τ · Δr`, a level move with mean-reversion shape), an inflation
process feeding the term premium at the long end (a slope move), and an
optional lognormal equity index. Simulated shifts are applied on top of
today's market curve. Every path is fully re-priced with a vectorised
pricer (interpolation is linear in the rate vector, so re-pricing all
instruments on all paths is a few matrix products — 10,000 paths in ~30 ms).
VaR is the loss quantile, ES the mean loss beyond it; both are reported by
factor, by asset class and against a key-rate-duration (delta) approximation.

## Stated simplifications

- Monte Carlo P&L is pure market risk: the settlement date does not roll
  forward, no carry or roll-down over the horizon (standard for a 1–10 day VaR).
- Model parameters (κ, θ, σ, correlations, pass-through) are inputs, not
  calibrated to history.
- No holiday calendars or business-day adjustment; coupon dates roll on
  calendar months.
- One equity factor shared by all equity positions.
- Sample curves in the dashboard are illustrative, not market data.

## What the tests prove

| Check | Value |
|---|---|
| 10y 5% semi-annual bond at 6% | 92.5613 (textbook) |
| Coupon = yield | price = par |
| FRN, zero spread, on the curve that projects it | exactly par |
| Zero-coupon Macaulay / convexity | T and T(T+1)/(1+y)² exactly |
| Analytic duration & convexity vs finite differences | agree to 1e-6 |
| Σ key-rate durations | = effective duration |
| Vectorised MC pricer vs reference pricer on random shifted curves | agree to 1e-9 |
| Simulated Vasicek mean & std vs closed form | within Monte Carlo error |
| Full-revaluation VaR vs delta VaR on a bond book | lower (convexity), within 2% at 10 days |
