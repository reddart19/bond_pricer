"""
instruments.py — The three instrument types.

Every instrument does ONE thing for the rest of the engine: it produces a list
of CashFlow objects (`instrument.cashflows(curve)`). Pricing, risk and Monte
Carlo never need to know whether they are looking at a fixed bond, a zero or
an FRN — they just discount cash flows. That is what makes the engine modular.

    FixedRateBond     : coupon = notional * coupon_rate / frequency, principal at maturity
    ZeroCouponBond    : one single cash flow (the notional) at maturity
    FloatingRateNote  : coupon = notional * (forward_rate + spread) * accrual
                        -> needs a curve to PROJECT the future coupons
"""
import calendar
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple

from curves import YieldCurve
from daycount import DayCount, year_fraction


# --------------------------------------------------------------------------
# Date helper
# --------------------------------------------------------------------------
def add_months(d: date, n: int) -> date:
    """Add n months (n may be negative), clamping to the end of month if needed.
    e.g. 31 Jan + 1 month -> 28/29 Feb."""
    m = d.month - 1 + n
    y = d.year + m // 12
    m = m % 12 + 1
    last_day = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, last_day))


# --------------------------------------------------------------------------
# Cash flow
# --------------------------------------------------------------------------
@dataclass
class CashFlow:
    pay_date: date        # when the money is paid
    start_date: date      # start of the accrual period (needed for accrued interest)
    t: float              # year fraction from settlement to pay_date
    coupon: float         # interest part
    principal: float      # capital part (0 except at maturity)
    rate: float = 0.0     # annualised coupon rate used for this period (info only)

    @property
    def total(self) -> float:
        return self.coupon + self.principal


# --------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------
@dataclass
class Instrument:
    notional: float
    settlement: date                  # pricing date
    maturity: date
    frequency: int = 1                # coupon payments per year: 1, 2, 4 or 12
    day_count: DayCount = DayCount.THIRTY_360
    name: str = ""

    def __post_init__(self):
        if self.notional <= 0:
            raise ValueError("notional must be positive")
        if self.maturity <= self.settlement:
            raise ValueError("maturity must be after settlement")
        if self.frequency not in (1, 2, 4, 12):
            raise ValueError("frequency must be 1, 2, 4 or 12")
        if isinstance(self.day_count, str) and not isinstance(self.day_count, DayCount):
            self.day_count = DayCount(self.day_count)
        if not self.name:
            self.name = f"{type(self).__name__} {self.maturity.isoformat()}"

    @property
    def years_to_maturity(self) -> float:
        return year_fraction(self.settlement, self.maturity, self.day_count)

    def schedule(self) -> List[Tuple[date, date]]:
        """(period_start, period_end) for every coupon period ending AFTER settlement.
        Periods are generated backwards from maturity so the last one lands exactly
        on the maturity date. The first period's start may be BEFORE settlement —
        that is normal (we are in the middle of a coupon period) and it is what
        accrued interest is computed from.
        """
        step = 12 // self.frequency
        periods = []
        k = 0
        while True:
            end = add_months(self.maturity, -k * step)
            if end <= self.settlement:
                break
            start = add_months(self.maturity, -(k + 1) * step)
            periods.append((start, end))
            k += 1
        periods.reverse()
        return periods

    def cashflows(self, curve: Optional[YieldCurve] = None) -> List[CashFlow]:
        raise NotImplementedError

    # convenience
    def _t(self, d: date) -> float:
        return year_fraction(self.settlement, d, self.day_count)


# --------------------------------------------------------------------------
# 1) Fixed-rate bullet bond
# --------------------------------------------------------------------------
@dataclass
class FixedRateBond(Instrument):
    coupon_rate: float = 0.0          # annual, decimal (0.05 = 5 %)

    def __post_init__(self):
        super().__post_init__()
        if self.coupon_rate < 0:
            raise ValueError("coupon_rate cannot be negative")

    def cashflows(self, curve: Optional[YieldCurve] = None) -> List[CashFlow]:
        periods = self.schedule()
        coupon = self.notional * self.coupon_rate / self.frequency
        flows = []
        for i, (start, end) in enumerate(periods):
            is_last = i == len(periods) - 1
            flows.append(CashFlow(
                pay_date=end,
                start_date=start,
                t=self._t(end),
                coupon=coupon,
                principal=self.notional if is_last else 0.0,
                rate=self.coupon_rate,
            ))
        return flows


# --------------------------------------------------------------------------
# 2) Zero-coupon bond
# --------------------------------------------------------------------------
@dataclass
class ZeroCouponBond(Instrument):
    """No coupons: `frequency` is only used as the compounding convention
    when quoting a yield (annual by default)."""

    def cashflows(self, curve: Optional[YieldCurve] = None) -> List[CashFlow]:
        return [CashFlow(
            pay_date=self.maturity,
            start_date=self.settlement,
            t=self._t(self.maturity),
            coupon=0.0,
            principal=self.notional,
            rate=0.0,
        )]


# --------------------------------------------------------------------------
# 3) Floating-rate note
# --------------------------------------------------------------------------
@dataclass
class FloatingRateNote(Instrument):
    """Coupon for each period = notional * (reference_rate + spread) * accrual.

    The reference rate for FUTURE periods is unknown today, so we use the
    forward rate implied by the curve (standard practice). The rate for the
    CURRENT period was already fixed at the last reset: pass it as
    `current_fixing`. If omitted, it is projected from the curve too.

    spread is in decimal: 0.0050 = 50 bp.
    """
    spread: float = 0.0
    current_fixing: Optional[float] = None
    day_count: DayCount = DayCount.ACT_360   # market convention for floaters

    def cashflows(self, curve: Optional[YieldCurve] = None) -> List[CashFlow]:
        if curve is None:
            raise ValueError("A FloatingRateNote needs a YieldCurve to project its coupons")

        periods = self.schedule()
        flows = []
        for i, (start, end) in enumerate(periods):
            t_start = max(self._t(start), 0.0)      # current period may have started in the past
            t_end = self._t(end)
            accrual = year_fraction(start, end, self.day_count)

            if i == 0 and self.current_fixing is not None:
                ref_rate = self.current_fixing
            else:
                ref_rate = curve.forward_rate(t_start, t_end)

            rate = ref_rate + self.spread
            is_last = i == len(periods) - 1
            flows.append(CashFlow(
                pay_date=end,
                start_date=start,
                t=t_end,
                coupon=self.notional * rate * accrual,
                principal=self.notional if is_last else 0.0,
                rate=rate,
            ))
        return flows
