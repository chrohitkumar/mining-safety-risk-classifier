"""
Generate a synthetic but domain-realistic underground mining shift dataset.

Each row = one shift at a mine section. Features are operational parameters
that a shift supervisor / safety officer would actually have on hand.
The target (incident_occurred) is driven by real risk relationships:
  - low ventilation rate + high methane -> much higher risk
  - older machinery -> higher risk
  - inexperienced workforce -> higher risk
  - longer shifts, greater depth -> higher risk
  - a recent incident in the last month -> higher risk (recency effect)
Noise is added so the signal is realistic, not a giveaway.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000  # number of shift-records

df = pd.DataFrame({
    "shift_duration_hrs": np.random.choice([8, 10, 12], size=N, p=[0.5, 0.3, 0.2]),
    "ventilation_rate_m3_per_min": np.round(np.random.normal(6.5, 1.8, N).clip(1.5, 12), 2),
    "methane_level_ppm": np.round(np.random.exponential(scale=150, size=N).clip(0, 1200), 1),
    "machinery_age_years": np.round(np.random.uniform(0, 20, N), 1),
    "avg_worker_experience_yrs": np.round(np.random.exponential(scale=5, size=N).clip(0, 30), 1),
    "workforce_size": np.random.randint(5, 40, N),
    "depth_m": np.round(np.random.uniform(50, 900, N), 0),
    "temperature_c": np.round(np.random.normal(28, 5, N), 1),
    "incidents_last_30_days": np.random.poisson(0.4, N),
})

# --- Build a realistic risk score from domain-driven relationships ---
risk_score = (
    -0.35 * (df["ventilation_rate_m3_per_min"] - 6.5)          # low ventilation -> risk up
    + 0.004 * df["methane_level_ppm"]                           # more methane -> risk up
    + 0.06 * df["machinery_age_years"]                          # older machines -> risk up
    - 0.09 * df["avg_worker_experience_yrs"]                    # experience -> risk down
    + 0.15 * (df["shift_duration_hrs"] - 8)                     # longer shifts -> risk up
    + 0.0015 * df["depth_m"]                                    # deeper -> risk up
    + 0.05 * (df["temperature_c"] - 28)                         # heat stress -> risk up
    + 0.9 * df["incidents_last_30_days"]                        # recent incidents -> risk up
    + np.random.normal(0, 1.0, N)                               # noise
)

# Convert risk score to a probability via logistic function, then sample outcome
prob_incident = 1 / (1 + np.exp(-(risk_score - 4.3)))
df["incident_occurred"] = np.random.binomial(1, prob_incident)

out_path = "/home/claude/mine_safety_risk/mine_shift_safety_data.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(f"Incident rate: {df['incident_occurred'].mean():.1%}")
print(df.head())
