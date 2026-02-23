"""Quick test of candlestick rendering."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate test data
rng = np.random.default_rng(42)
n = 50
returns = rng.normal(0, 0.001, n)
close = 22000.0 * np.cumprod(1 + returns)
high = close + np.abs(rng.normal(0, 10, n))
low = close - np.abs(rng.normal(0, 8, n))
op = close + rng.normal(0, 5, n)
high = np.maximum(high, np.maximum(op, close))
low = np.minimum(low, np.minimum(op, close))
vol = rng.poisson(5000, n)

start = datetime(2026, 1, 5, 9, 30)
dates = [start + timedelta(minutes=5*i) for i in range(n)]
bars = pd.DataFrame({
    'open': op, 'high': high, 'low': low, 'close': close, 'volume': vol
}, index=pd.DatetimeIndex(dates))

print(f"Data range: {bars['close'].min():.2f} - {bars['close'].max():.2f}")
print(f"Sample bar 0: O={bars['open'].iloc[0]:.2f} H={bars['high'].iloc[0]:.2f} "
      f"L={bars['low'].iloc[0]:.2f} C={bars['close'].iloc[0]:.2f}")

# Test both methods
fig = make_subplots(rows=2, cols=1, subplot_titles=("Method 1: Integer X", "Method 2: Datetime X"))

# Method 1: Integer x-values
x_int = list(range(n))
fig.add_trace(go.Candlestick(
    x=x_int,
    open=bars['open'].values,
    high=bars['high'].values,
    low=bars['low'].values,
    close=bars['close'].values,
    name="Integer X",
    increasing_line_color="#26a69a",
    decreasing_line_color="#ef5350",
), row=1, col=1)

# Method 2: Datetime x-values (original)
fig.add_trace(go.Candlestick(
    x=bars.index,
    open=bars['open'],
    high=bars['high'],
    low=bars['low'],
    close=bars['close'],
    name="Datetime X",
    increasing_line_color="#26a69a",
    decreasing_line_color="#ef5350",
), row=2, col=1)

fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
fig.write_html("test_chart_output.html")
print("\nChart saved to test_chart_output.html")
