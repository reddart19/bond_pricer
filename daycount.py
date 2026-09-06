"""
daycount.py — Day count conventions.

A day count convention answers one question: "how many YEARS separate two dates?"
Different markets use different rules, and the answer changes coupon accruals
and discount factors, so the engine makes it explicit everywhere.

    30/360    : every month has 30 days, every year 360 (US corporate / bond basis)
    ACT/360   : actual days elapsed, divided by 360 (money market, most FRNs)
    ACT/365   : actual days elapsed, divided by 365 (simple, used for curves)
"""
from datetime import date
from enum import Enum


class DayCount(str, Enum):
    THIRTY_360 = "30/360"
    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"


def year_fraction(start: date, end: date, convention: DayCount = DayCount.ACT_365) -> float:
    """Year fraction from `start` to `end` under `convention`. Returns 0.0 if end <= start."""
    if end <= start:
        return 0.0

    if convention == DayCount.ACT_365:
        return (end - start).days / 365.0

    if convention == DayCount.ACT_360:
        return (end - start).days / 360.0

    if convention == DayCount.THIRTY_360:
        d1 = min(start.day, 30)
        d2 = end.day
        if d1 == 30 and d2 == 31:
            d2 = 30
        days = 360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)
        return days / 360.0

    raise ValueError(f"Unknown day count convention: {convention}")
