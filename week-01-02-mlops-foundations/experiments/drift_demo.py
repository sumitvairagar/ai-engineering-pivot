from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
import pandas as pd
import numpy as np

np.random.seed(42)

# Reference = last month's normal market conditions
reference = pd.DataFrame({
    'order_size': np.random.normal(5000, 1000, 500),
    'market_volatility': np.random.normal(1.0, 0.3, 500),
    'days_since_last_trade': np.random.exponential(5, 500),
})

# Current = this week — market crash happened!
current = pd.DataFrame({
    'order_size': np.random.normal(7500, 2000, 500),      # bigger panic orders
    'market_volatility': np.random.normal(1.8, 0.5, 500),  # way more volatile
    'days_since_last_trade': np.random.exponential(5, 500), # this stayed the same
})

report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
snapshot = report.run(reference_data=reference, current_data=current)
snapshot.save_html("drift_report.html")
print("Done — open drift_report.html in your browser")
