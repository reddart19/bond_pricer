"""
test_app.py — Headless smoke test of the dashboard. Run with:  python test_app.py
Loads the app, runs a Monte Carlo, adds a position, loads a curve preset and
moves a scenario slider, checking that no exception is raised at any step.
"""
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=120).run()
assert not at.exception, at.exception[0].value
assert [t.label for t in at.tabs] == ["Book", "Pricing", "Curve", "Risk", "Scenarios", "Monte Carlo"]
print("  ✓ app loads, 6 tabs, book MV", [m.value for m in at.metric if m.label == "Book market value"][0])

[b for b in at.button if b.label == "Run simulation"][0].click().run()
assert not at.exception, at.exception[0].value
print("  ✓ Monte Carlo:", [(m.label, m.value) for m in at.metric if m.label in ("VaR 99%", "ES 99%")])

[b for b in at.button if b.label == "Add to book"][0].click().run()
assert not at.exception, at.exception[0].value
print("  ✓ position added, book MV", [m.value for m in at.metric if m.label == "Book market value"][0])

[b for b in at.button if b.label == "Load preset"][0].click().run()
assert not at.exception, at.exception[0].value
[s for s in at.slider if s.label == "Parallel (bp)"][0].set_value(-150).run()
assert not at.exception, at.exception[0].value
print("  ✓ preset loaded, scenario slider moved, P&L",
      [m.value for m in at.metric if m.label.startswith("P&L under")][0])
print("\nDashboard smoke test passed.")
