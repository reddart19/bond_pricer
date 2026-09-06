"""
curves.py — Zero-coupon yield curve.

The curve is the single object every other module prices off:
    - discount_factor(t)  : how much 1 € paid at time t is worth today
    - zero_rate(t)        : the continuously-compounded rate behind that factor
    - forward_rate(t1,t2) : the rate the market implies for the period [t1, t2]
                            (this is what a floating-rate note's coupon is set to)
    - shifted(...)        : returns a NEW curve with a shock applied — parallel,
                            steepener, key-rate, anything. Risk and Monte Carlo
                            modules are built on this.

Rates are stored continuously compounded:  DF(t) = exp(-r(t) * t).
To convert an annually-compounded market quote:  r_cc = ln(1 + r_annual).
Interpolation is linear on zero rates, flat extrapolation beyond the last tenor.
"""
import math
from dataclasses import dataclass
from typing import Callable, Sequence, Optional

import numpy as np


@dataclass
class YieldCurve:
    tenors: np.ndarray          # in years, strictly increasing, all > 0
    rates: np.ndarray           # continuously-compounded zero rates, decimal
    name: str = "curve"

    def __post_init__(self):
        self.tenors = np.asarray(self.tenors, dtype=float)
        self.rates = np.asarray(self.rates, dtype=float)
        if self.tenors.ndim != 1 or self.tenors.shape != self.rates.shape:
            raise ValueError("tenors and rates must be 1-D arrays of equal length")
        if self.tenors.size == 0:
            raise ValueError("a curve needs at least one point")
        if np.any(self.tenors <= 0):
            raise ValueError("tenors must be strictly positive")
        if np.any(np.diff(self.tenors) <= 0):
            raise ValueError("tenors must be strictly increasing")

    # ---------- constructors ----------
    @classmethod
    def flat(cls, rate: float, name: str = "flat") -> "YieldCurve":
        """A flat curve: same zero rate at every maturity."""
        return cls(np.array([1.0, 30.0]), np.array([rate, rate]), name)

    @classmethod
    def from_annual_rates(cls, tenors: Sequence[float], annual_rates: Sequence[float],
                          name: str = "curve") -> "YieldCurve":
        """Build from annually-compounded quotes (the way rates are usually shown)."""
        cc = np.log1p(np.asarray(annual_rates, dtype=float))
        return cls(np.asarray(tenors, dtype=float), cc, name)

    # ---------- core queries ----------
    def zero_rate(self, t):
        """Continuously-compounded zero rate at time t (scalar or array)."""
        return np.interp(t, self.tenors, self.rates)

    def discount_factor(self, t):
        """DF(t) = exp(-r(t) * t). Works on scalars and arrays."""
        t = np.asarray(t, dtype=float)
        return np.exp(-self.zero_rate(t) * t)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Simple (money-market style) forward rate for the period [t1, t2].
        f = (DF(t1)/DF(t2) - 1) / (t2 - t1)
        """
        if t2 <= t1:
            raise ValueError("forward_rate requires t2 > t1")
        df1 = float(self.discount_factor(t1))
        df2 = float(self.discount_factor(t2))
        return (df1 / df2 - 1.0) / (t2 - t1)

    def annual_rate(self, t):
        """Zero rate converted to annual compounding (for display)."""
        return np.expm1(self.zero_rate(t))

    # ---------- shocks ----------
    def shifted(self, shock: Callable[[np.ndarray], np.ndarray],
                name: Optional[str] = None) -> "YieldCurve":
        """Return a new curve whose rates are  rates + shock(tenors).
        `shock` receives the tenor array and returns the shift in DECIMAL
        at each tenor. This is the building block for every scenario.
        """
        new_rates = self.rates + np.asarray(shock(self.tenors), dtype=float)
        return YieldCurve(self.tenors.copy(), new_rates, name or f"{self.name} (shocked)")

    def parallel_shift(self, bp: float) -> "YieldCurve":
        """Every tenor moves by the same number of basis points."""
        return self.shifted(lambda t: np.full_like(t, bp / 10_000.0),
                            name=f"{self.name} {bp:+.0f}bp")

    def __repr__(self) -> str:
        pts = ", ".join(f"{t:g}y: {r*100:.2f}%" for t, r in zip(self.tenors, self.rates))
        return f"YieldCurve({self.name}: {pts})"
