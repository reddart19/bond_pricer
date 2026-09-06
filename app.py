"""
app.py — Fixed-income desk dashboard.

Run with:   streamlit run app.py     (or  python -m streamlit run app.py)

The sidebar holds the two things everything depends on: the settlement date
and the yield curve. The tabs are organised by question:
    Book        what do I hold?
    Pricing     what is one position worth, and why?
    Curve       what does the market say?
    Risk        how sensitive is the book, and to which part of the curve?
    Scenarios   what happens if the curve moves like this?
    Monte Carlo how bad can 10 days get?
"""
from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from curves import YieldCurve
from daycount import DayCount
from instruments import FixedRateBond, FloatingRateNote, ZeroCouponBond
from montecarlo import (Correlations, EquityPosition, InflationParams, MonteCarloEngine,
                        VasicekParams)
from portfolio import Portfolio, Position
from pricing import price_from_curve, pv_at_yield, solve_ytm
from risk import modified_duration, risk_metrics
from scenarios import butterfly, parallel, standard_set, steepener

# ==========================================================================
# Look & feel
# ==========================================================================
st.set_page_config(page_title="Fixed-income desk", page_icon="📈", layout="wide")

ACCENT, LOSS, GAIN, MUTED, SECOND = "#1F5F8B", "#B33A3A", "#2F7D5A", "#8A94A6", "#D98E3B"

FIXED, ZERO, FRN = "Fixed-rate bond", "Zero-coupon bond", "Floating-rate note"
FREQ_LABELS = {1: "Annual", 2: "Semi-annual", 4: "Quarterly", 12: "Monthly"}

CURVE_PRESETS = {
    "EUR sample (upward)": ([0.5, 1, 2, 5, 10, 30], [3.00, 3.20, 3.40, 3.70, 4.00, 4.20]),
    "USD sample (flat)":   ([0.5, 1, 2, 5, 10, 30], [4.00, 4.00, 4.00, 4.00, 4.00, 4.00]),
    "Inverted":            ([0.5, 1, 2, 5, 10, 30], [5.20, 5.00, 4.60, 4.20, 4.00, 3.90]),
    "Steep":               ([0.5, 1, 2, 5, 10, 30], [1.00, 1.30, 1.90, 2.90, 3.80, 4.40]),
}


def fmt(x, d=2):
    return f"{x:,.{d}f}"


# ==========================================================================
# State
# ==========================================================================
def demo_book(settlement: date):
    y = settlement.year
    return [
        dict(id=1, type=FIXED, name=f"5% Jan-{y + 9}", notional=100.0, maturity=date(y + 9, 1, 1),
             frequency=2, day_count="30/360", coupon=5.0, spread_bp=0.0, fixing=None, units=1000.0),
        dict(id=2, type=ZERO, name=f"Zero Jan-{y + 9}", notional=100.0, maturity=date(y + 9, 1, 1),
             frequency=1, day_count="30/360", coupon=0.0, spread_bp=0.0, fixing=None, units=500.0),
        dict(id=3, type=FRN, name="FRN 3M+50", notional=100.0, maturity=date(y + 4, 1, 1),
             frequency=4, day_count="ACT/360", coupon=0.0, spread_bp=50.0, fixing=None, units=800.0),
        dict(id=4, type=FIXED, name=f"3% Jan-{y + 1}", notional=100.0, maturity=date(y + 1, 1, 1),
             frequency=1, day_count="30/360", coupon=3.0, spread_bp=0.0, fixing=None, units=1200.0),
    ]


def init_state():
    ss = st.session_state
    if "settlement" not in ss:
        ss.settlement = date.today()
    if "positions" not in ss:
        ss.positions = demo_book(ss.settlement)
        ss.next_id = 5
    if "curve_df" not in ss:
        t, r = CURVE_PRESETS["EUR sample (upward)"]
        ss.curve_df = pd.DataFrame({"Tenor (years)": t, "Zero rate (%)": r}, dtype=float)
    if "mc_result" not in ss:
        ss.mc_result = None
    ss.setdefault("book_version", 0)
    ss.setdefault("curve_version", 0)


def build_instrument(spec, settlement):
    common = dict(notional=spec["notional"], settlement=settlement, maturity=spec["maturity"],
                  frequency=int(spec["frequency"]), day_count=DayCount(spec["day_count"]),
                  name=spec["name"])
    if spec["type"] == FIXED:
        return FixedRateBond(coupon_rate=spec["coupon"] / 100, **common)
    if spec["type"] == ZERO:
        return ZeroCouponBond(**common)
    fixing = spec["fixing"] / 100 if spec["fixing"] is not None else None
    return FloatingRateNote(spread=spec["spread_bp"] / 1e4, current_fixing=fixing, **common)


def build_curve(df: pd.DataFrame) -> YieldCurve:
    clean = df.dropna()
    clean = clean[clean["Tenor (years)"] > 0].sort_values("Tenor (years)")
    clean = clean.drop_duplicates("Tenor (years)")
    if len(clean) == 0:
        raise ValueError("The curve needs at least one tenor.")
    return YieldCurve.from_annual_rates(clean["Tenor (years)"].values,
                                        clean["Zero rate (%)"].values / 100, name="market")


def build_portfolio(specs, settlement):
    built, problems = [], []
    for s in specs:
        try:
            built.append((s, Position(build_instrument(s, settlement), s["units"])))
        except ValueError as e:
            problems.append(f"{s['name']}: {e}")
    return built, problems


# ==========================================================================
# Charts
# ==========================================================================
def chart_curve(curve: YieldCurve, shocked: YieldCurve = None):
    grid = np.linspace(0.25, float(curve.tenors[-1]), 120)
    rows = [{"Tenor": t, "Rate (%)": 100 * float(curve.annual_rate(t)), "Series": "Zero (base)"} for t in grid]
    rows += [{"Tenor": t, "Rate (%)": 100 * curve.forward_rate(t, t + 1), "Series": "1y forward (base)"}
             for t in grid if t + 1 <= curve.tenors[-1]]
    if shocked is not None:
        rows += [{"Tenor": t, "Rate (%)": 100 * float(shocked.annual_rate(t)), "Series": "Zero (shocked)"}
                 for t in grid]
    df = pd.DataFrame(rows)
    knots = pd.DataFrame({"Tenor": curve.tenors, "Rate (%)": 100 * curve.annual_rate(curve.tenors)})
    lines = alt.Chart(df).mark_line().encode(
        x=alt.X("Tenor:Q", title="Tenor (years)"),
        y=alt.Y("Rate (%):Q", scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N", scale=alt.Scale(domain=["Zero (base)", "1y forward (base)", "Zero (shocked)"],
                                                    range=[ACCENT, MUTED, LOSS]), legend=alt.Legend(title=None)),
        strokeDash=alt.condition(alt.datum.Series == "1y forward (base)", alt.value([4, 3]), alt.value([1, 0])),
    )
    pts = alt.Chart(knots).mark_point(color=ACCENT, filled=True, size=60).encode(x="Tenor:Q", y="Rate (%):Q")
    return (lines + pts).properties(height=300)


def chart_cashflows(cfs):
    df = pd.DataFrame([{"Date": cf.pay_date, "Amount": cf.coupon, "Part": "Coupon"} for cf in cfs] +
                      [{"Date": cf.pay_date, "Amount": cf.principal, "Part": "Principal"} for cf in cfs if cf.principal])
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Date:T", title=None), y=alt.Y("Amount:Q", title="Cash flow"),
        color=alt.Color("Part:N", scale=alt.Scale(domain=["Coupon", "Principal"], range=[ACCENT, MUTED]),
                        legend=alt.Legend(title=None)),
        tooltip=["Date:T", "Part:N", alt.Tooltip("Amount:Q", format=",.2f")],
    ).properties(height=220)


def chart_price_yield(inst, cfs, ytm, price):
    ys = np.linspace(0.001, 0.15, 150)
    f = inst.frequency
    prices = [pv_at_yield(cfs, y, f) for y in ys]
    dmod = modified_duration(cfs, ytm, f)
    tangent = [price * (1 - dmod * (y - ytm)) for y in ys]
    df = pd.DataFrame({"Yield (%)": 100 * ys, "Price": prices, "Duration tangent": tangent})
    long = df.melt("Yield (%)", var_name="Series", value_name="Value")
    lines = alt.Chart(long).mark_line().encode(
        x="Yield (%):Q", y=alt.Y("Value:Q", title="Dirty price", scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N", scale=alt.Scale(domain=["Price", "Duration tangent"], range=[ACCENT, SECOND]),
                        legend=alt.Legend(title=None)),
        strokeDash=alt.condition(alt.datum.Series == "Duration tangent", alt.value([4, 3]), alt.value([1, 0])),
    )
    pt = alt.Chart(pd.DataFrame({"Yield (%)": [100 * ytm], "Value": [price]})).mark_point(
        color=LOSS, filled=True, size=120).encode(x="Yield (%):Q", y="Value:Q")
    return (lines + pt).properties(height=300)


def chart_krd(pf: Portfolio, curve: YieldCurve, bump):
    w = pf.weights(curve)
    rows = []
    for wi, p in zip(w, pf.positions):
        m = risk_metrics(p.instrument, curve, bump)
        for k, v in m.key_rate_durations.items():
            rows.append({"Tenor": f"{k:g}y", "Position": p.name, "Contribution": wi * v, "order": k})
    df = pd.DataFrame(rows)
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Tenor:N", sort=alt.SortField("order"), title=None),
        y=alt.Y("Contribution:Q", title="Years of duration"),
        color=alt.Color("Position:N", legend=alt.Legend(title=None)),
        tooltip=["Position:N", "Tenor:N", alt.Tooltip("Contribution:Q", format=".3f")],
    ).properties(height=300)


def chart_scenarios(table: pd.DataFrame):
    long = table.melt("Scenario", value_vars=["P&L", "Taylor P&L"], var_name="Method", value_name="Value")
    long["Method"] = long["Method"].replace({"P&L": "Full repricing", "Taylor P&L": "Duration + convexity"})
    return alt.Chart(long).mark_bar().encode(
        y=alt.Y("Scenario:N", sort=None, title=None),
        x=alt.X("Value:Q", title="P&L"),
        color=alt.Color("Method:N", scale=alt.Scale(range=[ACCENT, SECOND]), legend=alt.Legend(title=None)),
        yOffset="Method:N",
        tooltip=["Scenario:N", "Method:N", alt.Tooltip("Value:Q", format=",.0f")],
    ).properties(height=26 * table["Scenario"].nunique() + 40)


def chart_ladder(ladder: pd.DataFrame):
    base = alt.Chart(ladder).encode(x=alt.X("bp:Q", title="Parallel shift (bp)"))
    area = base.mark_area(opacity=0.15, color=ACCENT).encode(y=alt.Y("P&L:Q", title="P&L"))
    line = base.mark_line(color=ACCENT).encode(y="P&L:Q")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED).encode(y="y:Q")
    return (area + line + zero).properties(height=260)


def chart_pnl_hist(res):
    df = pd.DataFrame({"P&L": res.pnl})
    v95, v99, e99 = res.var(res.pnl, 0.95), res.var(res.pnl, 0.99), res.es(res.pnl, 0.99)
    hist = alt.Chart(df).mark_bar(color=ACCENT, opacity=0.85).encode(
        x=alt.X("P&L:Q", bin=alt.Bin(maxbins=80), title="10-day P&L"),
        y=alt.Y("count():Q", title="Paths"),
    )
    marks = pd.DataFrame({"x": [-v95, -v99, -e99], "label": ["VaR 95%", "VaR 99%", "ES 99%"],
                          "dash": ["a", "b", "c"]})
    rules = alt.Chart(marks).mark_rule(strokeWidth=2).encode(
        x="x:Q", color=alt.Color("label:N", scale=alt.Scale(domain=["VaR 95%", "VaR 99%", "ES 99%"],
                                                            range=[SECOND, LOSS, "#6B1E1E"]),
                                 legend=alt.Legend(title=None, orient="top-left")))
    return (hist + rules).properties(height=320)


def chart_fan(res):
    fan = res.curve_fan().reset_index().rename(columns={"index": "Tenor"})
    fan["t"] = res.tenors
    base = alt.Chart(fan).encode(x=alt.X("t:Q", title="Tenor (years)"))
    outer = base.mark_area(opacity=0.18, color=ACCENT).encode(y=alt.Y("q1%:Q", title="Curve shift (bp)"), y2="q99%:Q")
    inner = base.mark_area(opacity=0.35, color=ACCENT).encode(y="q5%:Q", y2="q95%:Q")
    med = base.mark_line(color=ACCENT).encode(y="q50%:Q")
    return (outer + inner + med).properties(height=260, title="1%–99% and 5%–95% bands of the simulated curve shift")


def chart_paths(res, n=60):
    idx = np.arange(min(n, res.n_paths))
    days = np.arange(res.n_steps + 1)
    rows = [{"Day": d, "Path": int(i), "Short rate (%)": 100 * res.r_paths[i, d]} for i in idx for d in days]
    return alt.Chart(pd.DataFrame(rows)).mark_line(opacity=0.35, strokeWidth=1, color=ACCENT).encode(
        x="Day:Q", y=alt.Y("Short rate (%):Q", scale=alt.Scale(zero=False)), detail="Path:N",
    ).properties(height=260, title=f"{len(idx)} sample short-rate paths")


# ==========================================================================
# Sidebar
# ==========================================================================
init_state()
ss = st.session_state

with st.sidebar:
    st.title("Fixed-income desk")
    ss.settlement = st.date_input("Settlement date", value=ss.settlement)
    preset = st.selectbox("Curve preset", list(CURVE_PRESETS), index=0)
    if st.button("Load preset"):
        t, r = CURVE_PRESETS[preset]
        ss.curve_df = pd.DataFrame({"Tenor (years)": t, "Zero rate (%)": r}, dtype=float)
        ss.curve_version += 1
        ss.mc_result = None
    bump = st.slider("Bump for effective measures (bp)", 1, 50, 10)
    st.caption("Sample curves are illustrative, not market data. Edit any point in the Curve tab.")
    if st.button("Reset demo book"):
        ss.positions = demo_book(ss.settlement)
        ss.next_id = 5
        ss.book_version += 1
        ss.mc_result = None

try:
    curve = build_curve(ss.curve_df)
except ValueError as e:
    st.error(str(e))
    st.stop()

built, problems = build_portfolio(ss.positions, ss.settlement)
for p in problems:
    st.warning(f"Skipped {p}")
positions = [pos for _, pos in built]
built_ids = {s["id"] for s, _ in built}
pf = Portfolio(positions) if positions else None

tab_book, tab_price, tab_curve, tab_risk, tab_scen, tab_mc = st.tabs(
    ["Book", "Pricing", "Curve", "Risk", "Scenarios", "Monte Carlo"])

# ==========================================================================
# Book
# ==========================================================================
with tab_book:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Positions")
        if pf is None:
            st.info("The book is empty. Add a position on the right.")
        else:
            rows = []
            for s, pos in built:
                pr = price_from_curve(pos.instrument, curve)
                y = solve_ytm(pos.instrument, pr.dirty, is_clean=False, cashflows=pr.cashflows).ytm
                rows.append({"id": s["id"], "Name": s["name"], "Type": s["type"], "Maturity": s["maturity"],
                             "Units": s["units"], "Clean (per 100)": pr.clean_per_100, "YTM (%)": 100 * y,
                             "Market value": pos.units * pr.dirty})
            df = pd.DataFrame(rows)
            edited = st.data_editor(
                df, hide_index=True, num_rows="dynamic", key=f"book_editor_{ss.book_version}",
                column_config={
                    "id": None,
                    "Units": st.column_config.NumberColumn(min_value=0.0, step=100.0, format="%.0f"),
                    "Clean (per 100)": st.column_config.NumberColumn(format="%.4f", disabled=True),
                    "YTM (%)": st.column_config.NumberColumn(format="%.3f", disabled=True),
                    "Market value": st.column_config.NumberColumn(format="%,.0f", disabled=True),
                    "Type": st.column_config.TextColumn(disabled=True),
                    "Maturity": st.column_config.DateColumn(disabled=True),
                },
            )
            kept = {int(r["id"]): r for _, r in edited.dropna(subset=["id"]).iterrows()}
            new_specs = []
            for s in ss.positions:
                if s["id"] in kept:
                    row = kept[s["id"]]
                    units = s["units"] if pd.isna(row["Units"]) else float(row["Units"])
                    new_specs.append(dict(s, name=str(row["Name"]), units=units))
                elif s["id"] not in built_ids:
                    new_specs.append(s)             # invalid (skipped) rows stay in state
            if new_specs != ss.positions:
                ss.positions = new_specs
                ss.book_version += 1
                ss.mc_result = None
                st.rerun()
            st.metric("Book market value", fmt(pf.market_value(curve), 0))
            st.caption("Edit units or names in place; select a row and press Delete to remove it.")

    with right:
        st.subheader("Add a position")
        itype = st.selectbox("Instrument", [FIXED, ZERO, FRN])
        with st.form("add_position", clear_on_submit=False):
            c1, c2 = st.columns(2)
            name = c1.text_input("Name", value={FIXED: "New bond", ZERO: "New zero", FRN: "New FRN"}[itype])
            notional = c2.number_input("Notional", min_value=0.01, value=100.0, step=100.0)
            maturity = c1.date_input("Maturity", value=ss.settlement + timedelta(days=365 * 5))
            units = c2.number_input("Units", min_value=0.0, value=1000.0, step=100.0)
            freq = c1.selectbox("Coupon frequency", list(FREQ_LABELS), index={FIXED: 1, ZERO: 0, FRN: 2}[itype],
                                format_func=lambda k: FREQ_LABELS[k])
            dc = c2.selectbox("Day count", [d.value for d in DayCount], index={FIXED: 0, ZERO: 0, FRN: 1}[itype])
            coupon = spread_bp = 0.0
            fixing = None
            if itype == FIXED:
                coupon = st.number_input("Coupon rate (%)", min_value=0.0, value=4.0, step=0.25)
            elif itype == FRN:
                spread_bp = c1.number_input("Spread over reference (bp)", value=50.0, step=5.0)
                use_fix = c2.checkbox("Current coupon already fixed")
                fix_val = c2.number_input("Current fixing (%)", value=3.0, step=0.05)
                fixing = fix_val if use_fix else None
            if st.form_submit_button("Add to book"):
                ss.positions.append(dict(id=ss.next_id, type=itype, name=name, notional=notional,
                                         maturity=maturity, frequency=freq, day_count=dc, coupon=coupon,
                                         spread_bp=spread_bp, fixing=fixing, units=units))
                ss.next_id += 1
                ss.book_version += 1
                ss.mc_result = None
                st.rerun()

# ==========================================================================
# Pricing
# ==========================================================================
with tab_price:
    if pf is None:
        st.info("Add a position to price it.")
    else:
        names = [p.name for p in pf.positions]
        pick = st.selectbox("Position", names, key="price_pick")
        pos = pf.positions[names.index(pick)]
        inst = pos.instrument
        pr = price_from_curve(inst, curve)
        yres = solve_ytm(inst, pr.dirty, is_clean=False, cashflows=pr.cashflows)
        cfs = pr.cashflows

        m = st.columns(5)
        m[0].metric("Clean (per 100)", fmt(pr.clean_per_100, 4))
        m[1].metric("Dirty (per 100)", fmt(pr.dirty_per_100, 4))
        m[2].metric("Accrued (per 100)", fmt(100 * pr.accrued / inst.notional, 4))
        m[3].metric("Yield to maturity", f"{100 * yres.ytm:.3f} %")
        m[4].metric("Years to maturity", fmt(inst.years_to_maturity, 2))
        st.caption(f"YTM solved by {yres.method} in {yres.iterations} iterations, residual {yres.residual:.1e}. "
                   f"Compounding: {FREQ_LABELS[inst.frequency].lower()}, day count {inst.day_count.value}.")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Price against yield**")
            st.altair_chart(chart_price_yield(inst, cfs, yres.ytm, pr.dirty))
            st.caption("The dashed line is the duration approximation; the gap to the curve is convexity.")
        with c2:
            st.markdown("**Cash flow calendar**")
            st.altair_chart(chart_cashflows(cfs))
            cf_df = pd.DataFrame([{"Date": cf.pay_date, "Years": cf.t, "Rate (%)": 100 * cf.rate,
                                   "Coupon": cf.coupon, "Principal": cf.principal, "Total": cf.total,
                                   "Discount factor": float(curve.discount_factor(cf.t)),
                                   "Present value": cf.total * float(curve.discount_factor(cf.t))}
                                  for cf in cfs])
            st.dataframe(cf_df, hide_index=True, height=260,
                         column_config={c: st.column_config.NumberColumn(format="%.4f")
                                        for c in ["Years", "Rate (%)", "Discount factor"]} |
                                       {c: st.column_config.NumberColumn(format="%,.2f")
                                        for c in ["Coupon", "Principal", "Total", "Present value"]})

        with st.expander("Newton-Raphson convergence"):
            hist = pd.DataFrame({"Iteration": range(len(yres.history)),
                                 "Yield (%)": [100 * h for h in yres.history],
                                 "Price at yield": [pv_at_yield(cfs, h, inst.frequency) for h in yres.history]})
            st.dataframe(hist, hide_index=True,
                         column_config={"Yield (%)": st.column_config.NumberColumn(format="%.8f"),
                                        "Price at yield": st.column_config.NumberColumn(format="%.8f")})

# ==========================================================================
# Curve
# ==========================================================================
with tab_curve:
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.subheader("Zero rates")
        edited = st.data_editor(ss.curve_df, num_rows="dynamic", hide_index=True, key=f"curve_editor_{ss.curve_version}",
                                column_config={"Tenor (years)": st.column_config.NumberColumn(min_value=0.01, format="%.2f"),
                                               "Zero rate (%)": st.column_config.NumberColumn(format="%.3f", step=0.05)})
        if not edited.equals(ss.curve_df):
            ss.curve_df = edited
            ss.curve_version += 1
            ss.mc_result = None
            st.rerun()
        st.caption("Annually compounded, stored continuously compounded internally. "
                   "Linear interpolation between tenors, flat beyond.")
        dfs = pd.DataFrame({"Tenor": [f"{t:g}y" for t in curve.tenors],
                            "Discount factor": curve.discount_factor(curve.tenors),
                            "Zero cc (%)": 100 * curve.rates})
        st.dataframe(dfs, hide_index=True, column_config={
            "Discount factor": st.column_config.NumberColumn(format="%.6f"),
            "Zero cc (%)": st.column_config.NumberColumn(format="%.4f")})
    with c2:
        st.subheader("Zero and forward curves")
        st.altair_chart(chart_curve(curve))

# ==========================================================================
# Risk
# ==========================================================================
with tab_risk:
    if pf is None:
        st.info("Add a position to see its risk.")
    else:
        table = pf.risk_table(curve, bump)
        total = table.iloc[-1]
        m = st.columns(5)
        m[0].metric("Effective duration", fmt(total["Eff. duration"], 3))
        m[1].metric("Modified duration", fmt(total["Modified"], 3))
        m[2].metric("Convexity", fmt(total["Eff. convexity"], 2))
        m[3].metric("DV01", fmt(total["DV01"], 2))
        m[4].metric("Weighted YTM", f"{total['YTM %']:.3f} %")

        st.dataframe(table, hide_index=True, column_config={
            "Units": st.column_config.NumberColumn(format="%.0f"),
            "Clean (per 100)": st.column_config.NumberColumn(format="%.4f"),
            "Market value": st.column_config.NumberColumn(format="%,.0f"),
            "Weight": st.column_config.NumberColumn(format="%.1f%%"),
            "YTM %": st.column_config.NumberColumn(format="%.3f"),
            **{c: st.column_config.NumberColumn(format="%.3f")
               for c in ["Macaulay", "Modified", "Convexity", "DV01", "Eff. duration", "Eff. convexity"]},
        })
        st.caption("Analytic measures assume fixed cash flows at a flat yield; effective measures reprice the "
                   "position on a bumped curve. For a floating-rate note only the effective ones are meaningful.")

        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            st.markdown("**Key-rate duration by position** (contribution to book duration)")
            st.altair_chart(chart_krd(pf, curve, bump))
        with c2:
            st.markdown("**Key-rate durations**")
            krd = pf.key_rate_table(curve, bump)
            st.dataframe(krd, column_config={c: st.column_config.NumberColumn(format="%.3f") for c in krd.columns})

# ==========================================================================
# Scenarios
# ==========================================================================
with tab_scen:
    if pf is None:
        st.info("Add a position to run scenarios.")
    else:
        st.subheader("Build a shock")
        c1, c2, c3 = st.columns(3)
        p_bp = c1.slider("Parallel (bp)", -300, 300, 100, 5)
        s_bp = c2.slider("Steepener 2s10s (bp)", -150, 150, 0, 5)
        b_bp = c3.slider("Butterfly 2/10/30 (bp)", -100, 100, 0, 5)
        custom = parallel(p_bp) + steepener(s_bp) + butterfly(b_bp)
        shocked = custom.apply(curve)
        base_mv = pf.market_value(curve)
        r = pf.scenario_pnl(curve, custom)

        c1, c2 = st.columns([2, 1], gap="large")
        with c1:
            st.altair_chart(chart_curve(curve, shocked))
        with c2:
            st.metric("P&L under this shock", fmt(r["P&L"], 0), f"{100 * r['P&L %']:+.2f} %")
            per_pos = pd.DataFrame([{"Position": p.name,
                                     "P&L": p.market_value(shocked) - p.market_value(curve)}
                                    for p in pf.positions])
            st.dataframe(per_pos, hide_index=True,
                         column_config={"P&L": st.column_config.NumberColumn(format="%,.0f")})

        st.subheader("Standard scenario grid")
        table = pf.scenario_table(curve, standard_set(), bump)
        st.altair_chart(chart_scenarios(table))
        st.caption("Full repricing next to the duration-and-convexity estimate. They agree on parallel moves; "
                   "on steepeners and butterflies only the key-rate profile can explain the difference.")
        with st.expander("Scenario table"):
            st.dataframe(table, hide_index=True, column_config={
                "MV base": st.column_config.NumberColumn(format="%,.0f"),
                "MV shocked": st.column_config.NumberColumn(format="%,.0f"),
                "P&L": st.column_config.NumberColumn(format="%,.0f"),
                "P&L %": st.column_config.NumberColumn(format="%.2f%%"),
                "Avg shift bp": st.column_config.NumberColumn(format="%.1f"),
                "Taylor P&L": st.column_config.NumberColumn(format="%,.0f")})

        st.subheader("Parallel shock ladder")
        st.altair_chart(chart_ladder(pf.shock_ladder(curve)))

# ==========================================================================
# Monte Carlo
# ==========================================================================
with tab_mc:
    if pf is None:
        st.info("Add a position to simulate.")
    else:
        with st.form("mc_form"):
            st.subheader("Simulation")
            c1, c2, c3, c4 = st.columns(4)
            n_paths = c1.select_slider("Paths", [1_000, 5_000, 10_000, 25_000, 50_000, 100_000], value=10_000)
            horizon_days = c2.slider("Horizon (business days)", 1, 250, 10)
            seed = c3.number_input("Seed", value=42, step=1)
            r0_default = 100 * float(curve.zero_rate(curve.tenors[0]))

            st.markdown("**Short rate (Vasicek)**")
            c1, c2, c3 = st.columns(3)
            kappa = c1.slider("Mean reversion κ", 0.01, 1.0, 0.15, 0.01)
            sigma_bp = c2.slider("Volatility σ (bp / year)", 10, 300, 120, 5)
            theta = c3.number_input("Long-run level θ (%)", value=round(r0_default, 2), step=0.1)

            st.markdown("**Inflation**")
            c1, c2, c3 = st.columns(3)
            pi_sigma_bp = c1.slider("Inflation vol (bp / year)", 0, 300, 100, 5)
            pass_through = c2.slider("Pass-through to long yields", 0.0, 1.0, 0.5, 0.05)
            rho_ri = c3.slider("Corr(rates, inflation)", -0.9, 0.9, 0.4, 0.05)

            st.markdown("**Equity leg** (optional)")
            c1, c2, c3, c4 = st.columns(4)
            eq_on = c1.checkbox("Include an equity index", value=True)
            eq_mv = c2.number_input("Equity market value", min_value=0.0, value=50_000.0, step=5_000.0)
            eq_vol = c3.slider("Equity vol (annual)", 0.05, 0.60, 0.20, 0.01)
            rho_re = c4.slider("Corr(rates, equity)", -0.9, 0.9, -0.3, 0.05)
            run = st.form_submit_button("Run simulation")

        if run:
            try:
                eng = MonteCarloEngine(
                    pf, curve,
                    rates=VasicekParams(r0=r0_default / 100, kappa=kappa, theta=theta / 100, sigma=sigma_bp / 1e4),
                    inflation=InflationParams(sigma=pi_sigma_bp / 1e4, pass_through=pass_through),
                    equities=[EquityPosition("Equity index", units=eq_mv / 100, price=100.0, sigma=eq_vol)] if eq_on and eq_mv > 0 else [],
                    correlations=Correlations(rate_inflation=rho_ri, rate_equity=rho_re, inflation_equity=-0.2),
                )
                ss.mc_result = eng.simulate(n_paths=int(n_paths), horizon_years=horizon_days / 252, seed=int(seed))
            except ValueError as e:
                st.error(str(e))

        res = ss.mc_result
        if res is None:
            st.info("Set the parameters and run the simulation.")
        else:
            v95, v99 = res.var(res.pnl, 0.95), res.var(res.pnl, 0.99)
            e95, e99 = res.es(res.pnl, 0.95), res.es(res.pnl, 0.99)
            m = st.columns(5)
            m[0].metric("VaR 95%", fmt(v95, 0), f"{100 * v95 / res.mv0:.2f} % of MV", delta_color="inverse")
            m[1].metric("VaR 99%", fmt(v99, 0), f"{100 * v99 / res.mv0:.2f} % of MV", delta_color="inverse")
            m[2].metric("ES 95%", fmt(e95, 0))
            m[3].metric("ES 99%", fmt(e99, 0))
            m[4].metric("Paths", f"{res.n_paths:,}", f"{res.horizon_years * 252:.0f} days")

            st.altair_chart(chart_pnl_hist(res))
            c1, c2 = st.columns([3, 2], gap="large")
            with c1:
                st.markdown("**Loss measures by source**")
                st.dataframe(res.summary(), column_config={c: st.column_config.NumberColumn(format="%,.0f")
                                                            for c in res.summary().columns})
                st.caption("Delta uses the book's key-rate durations on the same paths — the gap to full "
                           "revaluation is convexity. Bonds and equity add to more than the total: diversification.")
            with c2:
                st.markdown("**Distribution**")
                stats = res.stats()
                st.dataframe(pd.DataFrame({"Value": [fmt(stats["mean P&L"], 0), fmt(stats["std P&L"], 0),
                                                     fmt(stats["worst path"], 0), fmt(stats["best path"], 0),
                                                     fmt(res.mv0, 0)]},
                                          index=["Mean P&L", "Std P&L", "Worst path", "Best path", "MV today"]))
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.altair_chart(chart_fan(res))
            with c2:
                st.altair_chart(chart_paths(res))
