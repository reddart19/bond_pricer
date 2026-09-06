"""
scenarios.py — Yield-curve shock scenarios.

A Scenario is just a function  tenors (years) -> shift in basis points.
Applying it to a curve returns a NEW curve (the original is never modified),
which every pricing/risk function can then consume unchanged.

Standard shapes (bp = size of the move):
    parallel(bp)     : every tenor moves by bp
    steepener(bp)    : short end -bp/2, long end +bp/2  (the 2s10s slope widens by bp)
    flattener(bp)    : the mirror image
    butterfly(bp)    : wings (2y, 30y) +bp, belly (10y) -bp  -> curvature shock
    custom({...})    : {tenor: bp} points, linearly interpolated, flat outside

Market jargon combines a level move with a slope move:
    bear_steepener   : rates up, long end up MORE      (inflation scare)
    bull_flattener   : rates down, long end down MORE  (flight to quality)
    bear_flattener   : rates up, short end up MORE     (central bank hikes)
    bull_steepener   : rates down, short end down MORE (central bank cuts)
"""
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from curves import YieldCurve


@dataclass
class Scenario:
    name: str
    shift_bp: Callable[[np.ndarray], np.ndarray]      # tenors -> bp at each tenor

    def apply(self, curve: YieldCurve) -> YieldCurve:
        return curve.shifted(lambda t: np.asarray(self.shift_bp(t), dtype=float) / 1e4,
                             name=f"{curve.name} | {self.name}")

    def describe(self, tenors) -> Dict[float, float]:
        """The shift in bp at the given tenors (for tables / plots)."""
        t = np.asarray(tenors, dtype=float)
        return {float(k): float(v) for k, v in zip(t, self.shift_bp(t))}

    def __add__(self, other: "Scenario") -> "Scenario":
        return Scenario(f"{self.name} + {other.name}",
                        lambda t: np.asarray(self.shift_bp(t)) + np.asarray(other.shift_bp(t)))


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def parallel(bp: float) -> Scenario:
    return Scenario(f"Parallel {bp:+.0f}bp", lambda t: np.full_like(t, bp, dtype=float))


def steepener(bp: float, short: float = 2.0, long: float = 10.0) -> Scenario:
    """Slope between `short` and `long` widens by bp, centred (short -bp/2, long +bp/2).
    Flat beyond the anchors."""
    def f(t):
        w = np.clip((np.asarray(t, dtype=float) - short) / (long - short), 0.0, 1.0)
        return -bp / 2.0 + bp * w
    return Scenario(f"Steepener {bp:+.0f}bp ({short:g}s{long:g}s)", f)


def flattener(bp: float, short: float = 2.0, long: float = 10.0) -> Scenario:
    s = steepener(-bp, short, long)
    return Scenario(f"Flattener {bp:+.0f}bp ({short:g}s{long:g}s)", s.shift_bp)


def butterfly(bp: float, short: float = 2.0, belly: float = 10.0, long: float = 30.0) -> Scenario:
    """Wings +bp, belly -bp. Positive bp = 'negative butterfly' in trader talk
    (belly outperforms). Use a negative bp for the opposite."""
    def f(t):
        return np.interp(np.asarray(t, dtype=float), [short, belly, long], [bp, -bp, bp])
    return Scenario(f"Butterfly {bp:+.0f}bp ({short:g}/{belly:g}/{long:g})", f)


def custom(points: Dict[float, float], name: str = "Custom") -> Scenario:
    """points = {tenor: bp}. Linear interpolation between points, flat outside."""
    ks = sorted(points)
    vs = [points[k] for k in ks]
    return Scenario(name, lambda t: np.interp(np.asarray(t, dtype=float), ks, vs))


# --------------------------------------------------------------------------
# Market-standard combinations
# --------------------------------------------------------------------------
def bear_steepener(level_bp: float = 50.0, slope_bp: float = 50.0) -> Scenario:
    s = parallel(level_bp) + steepener(slope_bp)
    return Scenario(f"Bear steepener ({level_bp:+.0f}/{slope_bp:+.0f})", s.shift_bp)


def bull_flattener(level_bp: float = 50.0, slope_bp: float = 50.0) -> Scenario:
    s = parallel(-level_bp) + flattener(slope_bp)
    return Scenario(f"Bull flattener ({-level_bp:+.0f}/{slope_bp:+.0f})", s.shift_bp)


def bear_flattener(level_bp: float = 50.0, slope_bp: float = 50.0) -> Scenario:
    s = parallel(level_bp) + flattener(slope_bp)
    return Scenario(f"Bear flattener ({level_bp:+.0f}/{slope_bp:+.0f})", s.shift_bp)


def bull_steepener(level_bp: float = 50.0, slope_bp: float = 50.0) -> Scenario:
    s = parallel(-level_bp) + steepener(slope_bp)
    return Scenario(f"Bull steepener ({-level_bp:+.0f}/{slope_bp:+.0f})", s.shift_bp)


def standard_set() -> List[Scenario]:
    """The usual grid a risk desk looks at every morning."""
    return [
        parallel(-200), parallel(-100), parallel(-50), parallel(+50), parallel(+100), parallel(+200),
        steepener(50), flattener(50), butterfly(25), butterfly(-25),
        bear_steepener(), bull_flattener(), bear_flattener(), bull_steepener(),
    ]
