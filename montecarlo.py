"""
montecarlo.py — Monte Carlo Value-at-Risk and Expected Shortfall.

THE QUESTION
    "Over the next H days, how much can this portfolio lose, and how bad is
    the average of the worst cases?"  VaR_99 answers the first (the loss that
    is only exceeded 1% of the time); ES_99 answers the second (the mean loss
    GIVEN that we are in that 1% tail). ES is the regulatory measure (FRTB)
    because VaR says nothing about how deep the tail goes.

THE RISK FACTORS  (all simulated jointly with correlated Gaussian shocks)
    1. Short rate r_t      — Vasicek / Ornstein-Uhlenbeck:
                             dr = kappa (theta - r) dt + sigma dW
         Under Vasicek the whole zero curve is affine in r, so a move dr
         shifts the yield at tenor tau by  B(tau)/tau * dr  with
         B(tau) = (1 - e^{-kappa tau}) / kappa.  Short end moves fully,
         long end less (mean reversion)  ->  a LEVEL shock with a shape.
    2. Inflation pi_t      — a second OU process. An inflation surprise feeds
                             into the term premium at the long end:
                             shift(tau) += pass_through * dpi * (1 - e^{-tau/T})
                             -> a SLOPE shock. Rates and inflation are correlated.
    3. Equity (optional)   — lognormal index correlated with both, so a bond
                             book can be stress-tested next to an equity leg.
    The simulated shifts are applied ON TOP of today's market curve, so
    today's prices are reproduced exactly and only the DYNAMICS come from
    the model.

THE VALUATION
    Every path is FULLY REPRICED (no duration approximation) with a
    vectorised pricer: all cash flows of all instruments on all paths in a
    handful of matrix operations, so 10,000 paths take well under a second.
    A key-rate-duration (delta) approximation is computed on the same paths
    for comparison — the gap between the two is the convexity effect.

SIMPLIFICATIONS (stated, as any risk engine should)
    - P&L is pure market risk: curve moves, settlement date does not roll,
      no carry / roll-down. Standard for a 1-10 day VaR.
    - Exact OU transition is used per step, so the discretisation is exact.
    - Parameters are inputs (kappa, theta, sigma...). Calibrating them to
      history is a separate exercise.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from curves import YieldCurve
from daycount import year_fraction
from instruments import FloatingRateNote, Instrument
from portfolio import Portfolio


# ==========================================================================
# 1. Vectorised repricing
# ==========================================================================
def _interp_weights(t: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    """Matrix W such that  np.interp(t, tenors, v) == W @ v  for ANY vector v.
    Linear interpolation is linear in v, which is exactly what lets us
    reprice thousands of shifted curves with one matrix product."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n_t, n_k = len(t), len(tenors)
    W = np.zeros((n_t, n_k))
    if n_k == 1:
        W[:, 0] = 1.0
        return W
    j = np.searchsorted(tenors, t, side="right") - 1
    for i, (ti, ji) in enumerate(zip(t, j)):
        if ji < 0:
            W[i, 0] = 1.0
        elif ji >= n_k - 1:
            W[i, -1] = 1.0
        else:
            w = (tenors[ji + 1] - ti) / (tenors[ji + 1] - tenors[ji])
            W[i, ji], W[i, ji + 1] = w, 1.0 - w
    return W


class PathPricer:
    """Pre-computes an instrument's cash-flow times so it can be repriced on
    many shifted curves at once.  price(shifts) with shifts of shape
    (n_paths, n_tenors) returns dirty prices of shape (n_paths,)."""

    def __init__(self, inst: Instrument, curve: YieldCurve):
        self.inst = inst
        cfs = inst.cashflows(curve)
        self.t_end = np.array([cf.t for cf in cfs])
        self.base_r_end = np.atleast_1d(curve.zero_rate(self.t_end))
        self.W_end = _interp_weights(self.t_end, curve.tenors)
        self.is_frn = isinstance(inst, FloatingRateNote)
        if self.is_frn:
            self.t_start = np.array([max(inst._t(cf.start_date), 0.0) for cf in cfs])
            self.accrual = np.array([year_fraction(cf.start_date, cf.pay_date, inst.day_count)
                                     for cf in cfs])
            self.base_r_start = np.atleast_1d(curve.zero_rate(self.t_start))
            self.W_start = _interp_weights(self.t_start, curve.tenors)
            self.fixed_first = inst.current_fixing is not None
            self.first_coupon = cfs[0].coupon
        else:
            self.amount = np.array([cf.total for cf in cfs])

    def price(self, shifts: np.ndarray) -> np.ndarray:
        shifts = np.atleast_2d(shifts)
        r_end = self.base_r_end[None, :] + shifts @ self.W_end.T
        df_end = np.exp(-r_end * self.t_end[None, :])
        if not self.is_frn:
            return df_end @ self.amount
        r_start = self.base_r_start[None, :] + shifts @ self.W_start.T
        df_start = np.exp(-r_start * self.t_start[None, :])
        fwd = (df_start / df_end - 1.0) / (self.t_end - self.t_start)[None, :]
        coupon = self.inst.notional * (fwd + self.inst.spread) * self.accrual[None, :]
        if self.fixed_first:
            coupon[:, 0] = self.first_coupon
        return (coupon * df_end).sum(axis=1) + df_end[:, -1] * self.inst.notional


# ==========================================================================
# 2. Risk-factor models
# ==========================================================================
@dataclass
class VasicekParams:
    r0: float                       # today's short rate
    kappa: float = 0.15             # mean-reversion speed (per year)
    theta: Optional[float] = None   # long-run level (defaults to r0: no drift)
    sigma: float = 0.012            # annualised vol of the short rate (0.012 = 120 bp)

    def __post_init__(self):
        if self.theta is None:
            self.theta = self.r0
        if self.kappa <= 0 or self.sigma < 0:
            raise ValueError("kappa must be > 0 and sigma >= 0")

    @classmethod
    def from_curve(cls, curve: YieldCurve, **kw) -> "VasicekParams":
        return cls(r0=float(curve.zero_rate(curve.tenors[0])), **kw)

    def loading(self, tenors) -> np.ndarray:
        """B(tau)/tau: how much the zero rate at each tenor moves per unit dr."""
        t = np.asarray(tenors, dtype=float)
        return (1.0 - np.exp(-self.kappa * t)) / (self.kappa * t)


@dataclass
class InflationParams:
    pi0: float = 0.02
    kappa: float = 0.5
    theta: float = 0.02
    sigma: float = 0.010            # annualised vol of inflation expectations
    pass_through: float = 0.5       # fraction of an inflation surprise priced into long yields
    loading_tenor: float = 5.0      # loading = pass_through * (1 - exp(-tau / loading_tenor))

    def loading(self, tenors) -> np.ndarray:
        t = np.asarray(tenors, dtype=float)
        return self.pass_through * (1.0 - np.exp(-t / self.loading_tenor))


@dataclass
class EquityPosition:
    name: str
    units: float
    price: float
    mu: float = 0.06                # annual drift
    sigma: float = 0.18             # annual vol

    @property
    def market_value(self) -> float:
        return self.units * self.price


@dataclass
class Correlations:
    rate_inflation: float = 0.4
    rate_equity: float = -0.3
    inflation_equity: float = -0.2

    def matrix(self) -> np.ndarray:
        c = np.array([[1.0, self.rate_inflation, self.rate_equity],
                      [self.rate_inflation, 1.0, self.inflation_equity],
                      [self.rate_equity, self.inflation_equity, 1.0]])
        if np.any(np.linalg.eigvalsh(c) <= 0):
            raise ValueError("correlation matrix is not positive definite")
        return c


def ou_step(x: np.ndarray, kappa: float, theta: float, sigma: float,
            dt: float, z: np.ndarray) -> np.ndarray:
    """Exact one-step transition of dX = kappa (theta - X) dt + sigma dW."""
    e = np.exp(-kappa * dt)
    std = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))
    return x * e + theta * (1.0 - e) + std * z


# ==========================================================================
# 3. Results
# ==========================================================================
@dataclass
class MCResult:
    n_paths: int
    horizon_years: float
    n_steps: int
    tenors: np.ndarray
    mv0: float                       # total market value today (bonds + equity)
    mv0_bonds: float
    mv0_equity: float
    pnl: np.ndarray                  # total P&L per path (full revaluation)
    pnl_bonds: np.ndarray
    pnl_equity: np.ndarray
    pnl_delta_bonds: np.ndarray      # key-rate-duration (linear) approximation
    pnl_rates_only: np.ndarray       # bonds repriced with the rate factor only
    pnl_inflation_only: np.ndarray   # bonds repriced with the inflation factor only
    shifts: np.ndarray               # (n_paths, n_tenors) curve shifts, decimal
    r_paths: np.ndarray              # (n_paths, n_steps+1)
    pi_paths: np.ndarray
    seed: Optional[int] = None

    # ---- tail measures ----
    @staticmethod
    def var(pnl: np.ndarray, alpha: float = 0.99) -> float:
        """Loss (positive number) exceeded with probability 1 - alpha."""
        return float(-np.quantile(pnl, 1.0 - alpha))

    @staticmethod
    def es(pnl: np.ndarray, alpha: float = 0.99) -> float:
        """Mean loss in the (1 - alpha) tail."""
        v = MCResult.var(pnl, alpha)
        tail = pnl[pnl <= -v]
        return float(-tail.mean()) if tail.size else v

    @property
    def pnl_delta(self) -> np.ndarray:
        return self.pnl_delta_bonds + self.pnl_equity

    def summary(self, alphas: Sequence[float] = (0.95, 0.99)) -> pd.DataFrame:
        cols = {
            "Full revaluation": self.pnl,
            "Delta (KRD) approx": self.pnl_delta,
            "Bonds only": self.pnl_bonds,
            "Equity only": self.pnl_equity,
            "Rates factor only": self.pnl_rates_only,
            "Inflation factor only": self.pnl_inflation_only,
        }
        rows = {}
        for a in alphas:
            rows[f"VaR {a:.0%}"] = {k: self.var(v, a) for k, v in cols.items()}
            rows[f"ES {a:.0%}"] = {k: self.es(v, a) for k, v in cols.items()}
        df = pd.DataFrame(rows).T
        if self.mv0_equity == 0:
            df = df.drop(columns=["Equity only"])
        return df

    def stats(self) -> Dict[str, float]:
        return {
            "paths": self.n_paths,
            "horizon_days": self.horizon_years * 252,
            "MV today": self.mv0,
            "mean P&L": float(self.pnl.mean()),
            "std P&L": float(self.pnl.std()),
            "worst path": float(self.pnl.min()),
            "best path": float(self.pnl.max()),
            "VaR 99% (% of MV)": self.var(self.pnl, 0.99) / self.mv0,
            "ES 99% (% of MV)": self.es(self.pnl, 0.99) / self.mv0,
        }

    def curve_fan(self, quantiles: Sequence[float] = (0.01, 0.05, 0.5, 0.95, 0.99)) -> pd.DataFrame:
        """Quantiles of the simulated curve shift at each tenor, in bp."""
        q = np.quantile(self.shifts * 1e4, quantiles, axis=0)
        return pd.DataFrame(q.T, index=[f"{t:g}y" for t in self.tenors],
                            columns=[f"q{p:.0%}" for p in quantiles])


# ==========================================================================
# 4. Engine
# ==========================================================================
class MonteCarloEngine:
    def __init__(self, portfolio: Portfolio, curve: YieldCurve,
                 rates: Optional[VasicekParams] = None,
                 inflation: Optional[InflationParams] = None,
                 equities: Optional[List[EquityPosition]] = None,
                 correlations: Optional[Correlations] = None):
        self.portfolio = portfolio
        self.curve = curve
        self.rates = rates or VasicekParams.from_curve(curve)
        self.inflation = inflation or InflationParams()
        self.equities = equities or []
        self.corr = correlations or Correlations()

        # pre-compute everything that does not depend on the paths
        self.pricers = [PathPricer(p.instrument, curve) for p in portfolio.positions]
        self.units = np.array([p.units for p in portfolio.positions])
        zero = np.zeros((1, len(curve.tenors)))
        self.mv0_bonds = float(sum(u * pr.price(zero)[0] for u, pr in zip(self.units, self.pricers)))
        self.mv0_equity = float(sum(e.market_value for e in self.equities))
        krd = portfolio.key_rate_durations(curve)
        self.krd_vec = np.array([krd[float(t)] for t in curve.tenors])
        self.rate_loading = self.rates.loading(curve.tenors)
        self.infl_loading = self.inflation.loading(curve.tenors)

    def _reprice_bonds(self, shifts: np.ndarray) -> np.ndarray:
        mv = np.zeros(shifts.shape[0])
        for u, pr in zip(self.units, self.pricers):
            mv += u * pr.price(shifts)
        return mv

    def simulate(self, n_paths: int = 10_000, horizon_years: float = 10 / 252,
                 n_steps: Optional[int] = None, seed: Optional[int] = 42) -> MCResult:
        n_steps = n_steps or max(1, int(round(horizon_years * 252)))   # daily steps
        dt = horizon_years / n_steps
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(self.corr.matrix())

        r = np.full(n_paths, self.rates.r0)
        pi = np.full(n_paths, self.inflation.pi0)
        z_eq_cum = np.zeros(n_paths)                     # cumulated equity shock (sqrt(dt)-scaled)
        r_paths = np.empty((n_paths, n_steps + 1)); r_paths[:, 0] = r
        pi_paths = np.empty((n_paths, n_steps + 1)); pi_paths[:, 0] = pi

        for k in range(1, n_steps + 1):
            z = rng.standard_normal((n_paths, 3)) @ L.T   # correlated N(0,1) shocks
            r = ou_step(r, self.rates.kappa, self.rates.theta, self.rates.sigma, dt, z[:, 0])
            pi = ou_step(pi, self.inflation.kappa, self.inflation.theta, self.inflation.sigma, dt, z[:, 1])
            z_eq_cum += np.sqrt(dt) * z[:, 2]
            r_paths[:, k], pi_paths[:, k] = r, pi

        # ---- curve shifts at the horizon ----
        dr = r - self.rates.r0
        dpi = pi - self.inflation.pi0
        shifts_rate = dr[:, None] * self.rate_loading[None, :]
        shifts_infl = dpi[:, None] * self.infl_loading[None, :]
        shifts = shifts_rate + shifts_infl

        # ---- full revaluation ----
        pnl_bonds = self._reprice_bonds(shifts) - self.mv0_bonds
        pnl_rates = self._reprice_bonds(shifts_rate) - self.mv0_bonds
        pnl_infl = self._reprice_bonds(shifts_infl) - self.mv0_bonds
        pnl_delta = -self.mv0_bonds * (shifts @ self.krd_vec)

        pnl_equity = np.zeros(n_paths)
        for e in self.equities:
            log_ret = (e.mu - 0.5 * e.sigma ** 2) * horizon_years + e.sigma * z_eq_cum
            pnl_equity += e.units * e.price * (np.exp(log_ret) - 1.0)

        return MCResult(
            n_paths=n_paths, horizon_years=horizon_years, n_steps=n_steps,
            tenors=self.curve.tenors.copy(),
            mv0=self.mv0_bonds + self.mv0_equity, mv0_bonds=self.mv0_bonds,
            mv0_equity=self.mv0_equity,
            pnl=pnl_bonds + pnl_equity, pnl_bonds=pnl_bonds, pnl_equity=pnl_equity,
            pnl_delta_bonds=pnl_delta, pnl_rates_only=pnl_rates, pnl_inflation_only=pnl_infl,
            shifts=shifts, r_paths=r_paths, pi_paths=pi_paths, seed=seed,
        )
