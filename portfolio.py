"""
portfolio.py — Aggregating positions.

A Position is an instrument and a number of units held. Market value is
units * dirty price. Portfolio-level measures follow two rules:
    - durations and convexities are MARKET-VALUE-WEIGHTED averages
    - DV01 and P&L are simply ADDITIVE
Scenario P&L is computed by FULL REPRICING on the shocked curve — the
duration/convexity Taylor approximation is reported next to it so you can
see where the approximation breaks down (large or non-parallel moves).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from curves import YieldCurve
from instruments import Instrument
from pricing import price_from_curve
from risk import RiskMetrics, risk_metrics, taylor_price_change
from scenarios import Scenario, parallel


@dataclass
class Position:
    instrument: Instrument
    units: float = 1.0

    def market_value(self, curve: YieldCurve) -> float:
        return self.units * price_from_curve(self.instrument, curve).dirty

    @property
    def name(self) -> str:
        return self.instrument.name


class Portfolio:
    def __init__(self, positions: List[Position], name: str = "Portfolio"):
        if not positions:
            raise ValueError("a portfolio needs at least one position")
        self.positions = positions
        self.name = name

    # ---------- valuation ----------
    def market_value(self, curve: YieldCurve) -> float:
        return sum(p.market_value(curve) for p in self.positions)

    def weights(self, curve: YieldCurve) -> np.ndarray:
        mv = np.array([p.market_value(curve) for p in self.positions])
        return mv / mv.sum()

    # ---------- risk ----------
    def position_metrics(self, curve: YieldCurve, bump_bp: float = 10.0) -> List[RiskMetrics]:
        return [risk_metrics(p.instrument, curve, bump_bp) for p in self.positions]

    def risk_table(self, curve: YieldCurve, bump_bp: float = 10.0) -> pd.DataFrame:
        rows = []
        metrics = self.position_metrics(curve, bump_bp)
        total_mv = self.market_value(curve)
        for pos, m in zip(self.positions, metrics):
            mv = pos.units * m.price
            rows.append({
                "Instrument": pos.name,
                "Type": type(pos.instrument).__name__,
                "Units": pos.units,
                "Clean (per 100)": 100 * m.clean / pos.instrument.notional,
                "Market value": mv,
                "Weight": mv / total_mv,
                "YTM %": 100 * m.ytm,
                "Macaulay": m.macaulay,
                "Modified": m.modified,
                "Convexity": m.convexity,
                "DV01": pos.units * m.dv01,
                "Eff. duration": m.effective_duration,
                "Eff. convexity": m.effective_convexity,
            })
        df = pd.DataFrame(rows)
        w = df["Weight"].values
        total = {
            "Instrument": "TOTAL", "Type": "", "Units": df["Units"].sum(),
            "Clean (per 100)": np.nan, "Market value": total_mv, "Weight": 1.0,
            "YTM %": float(np.dot(w, df["YTM %"])),
            "Macaulay": float(np.dot(w, df["Macaulay"])),
            "Modified": float(np.dot(w, df["Modified"])),
            "Convexity": float(np.dot(w, df["Convexity"])),
            "DV01": df["DV01"].sum(),
            "Eff. duration": float(np.dot(w, df["Eff. duration"])),
            "Eff. convexity": float(np.dot(w, df["Eff. convexity"])),
        }
        return pd.concat([df, pd.DataFrame([total])], ignore_index=True)

    def duration(self, curve: YieldCurve, effective: bool = True, bump_bp: float = 10.0) -> float:
        w = self.weights(curve)
        ms = self.position_metrics(curve, bump_bp)
        d = [m.effective_duration if effective else m.modified for m in ms]
        return float(np.dot(w, d))

    def convexity(self, curve: YieldCurve, effective: bool = True, bump_bp: float = 10.0) -> float:
        w = self.weights(curve)
        ms = self.position_metrics(curve, bump_bp)
        c = [m.effective_convexity if effective else m.convexity for m in ms]
        return float(np.dot(w, c))

    def dv01(self, curve: YieldCurve) -> float:
        return sum(p.units * risk_metrics(p.instrument, curve, with_krd=False).dv01
                   for p in self.positions)

    def key_rate_durations(self, curve: YieldCurve, bump_bp: float = 10.0) -> Dict[float, float]:
        """Market-value-weighted key-rate durations of the whole book."""
        w = self.weights(curve)
        ms = self.position_metrics(curve, bump_bp)
        out = {float(k): 0.0 for k in curve.tenors}
        for wi, m in zip(w, ms):
            for k, v in m.key_rate_durations.items():
                out[k] += wi * v
        return out

    def key_rate_table(self, curve: YieldCurve, bump_bp: float = 10.0) -> pd.DataFrame:
        ms = self.position_metrics(curve, bump_bp)
        df = pd.DataFrame([m.key_rate_durations for m in ms],
                          index=[p.name for p in self.positions])
        df.columns = [f"{c:g}y" for c in df.columns]
        w = self.weights(curve)
        df.loc["PORTFOLIO"] = w @ df.values
        df["Sum"] = df.sum(axis=1)
        return df

    # ---------- scenarios ----------
    def scenario_pnl(self, curve: YieldCurve, scenario: Scenario) -> dict:
        base = self.market_value(curve)
        shocked = self.market_value(scenario.apply(curve))
        return {"Scenario": scenario.name, "MV base": base, "MV shocked": shocked,
                "P&L": shocked - base, "P&L %": (shocked - base) / base}

    def scenario_table(self, curve: YieldCurve, scenarios: List[Scenario],
                       bump_bp: float = 10.0) -> pd.DataFrame:
        """Full-repricing P&L per scenario, plus the duration/convexity Taylor
        estimate using the scenario's AVERAGE shift (only meaningful for
        parallel-ish moves — that gap is the whole point of key-rate durations)."""
        base = self.market_value(curve)
        d = self.duration(curve, effective=True, bump_bp=bump_bp)
        c = self.convexity(curve, effective=True, bump_bp=bump_bp)
        rows = []
        for sc in scenarios:
            r = self.scenario_pnl(curve, sc)
            avg_bp = float(np.mean(list(sc.describe(curve.tenors).values())))
            r["Avg shift bp"] = avg_bp
            r["Taylor P&L"] = taylor_price_change(base, d, c, avg_bp / 1e4)
            rows.append(r)
        return pd.DataFrame(rows)

    def shock_ladder(self, curve: YieldCurve,
                     bps: Optional[List[float]] = None) -> pd.DataFrame:
        """P&L for a ladder of parallel shocks (for the classic 'ladder' chart)."""
        bps = bps if bps is not None else list(range(-300, 301, 25))
        return pd.DataFrame([self.scenario_pnl(curve, parallel(b)) | {"bp": b} for b in bps])
