# Mine Shift Safety Risk Prediction

A simple machine learning project that predicts whether an underground
mining shift is likely to have a safety incident, using operational
parameters a shift supervisor already has on hand.

## 1. The problem

> "Mine safety incidents don't happen randomly — they cluster around specific
> conditions: poor ventilation, high methane, old machinery, inexperienced crews.
> I built a model that scores each shift's risk *before* it starts, so a safety
> officer can intervene — extra ventilation checks, pairing junior workers with
> experienced ones, shortening a shift — instead of reacting after an incident."

## 2. Data

Since real incident data is confidential, I generated a **synthetic dataset
of 2,000 shifts** using realistic domain relationships from mine ventilation
and safety engineering (Hardy-Cross/Atkinson-style ventilation logic, methane
accumulation, fatigue from shift length, workforce experience). This is a
standard and honest approach for a portfolio project — be upfront in the
interview that the data is synthetic but the *relationships* are grounded in
real mining safety principles, not made up.

**Features (9):**
| Feature | Why it matters |
|---|---|
| `ventilation_rate_m3_per_min` | Poor ventilation → gas/dust accumulation |
| `methane_level_ppm` | Direct explosion/asphyxiation risk |
| `machinery_age_years` | Older equipment fails more often |
| `avg_worker_experience_yrs` | Experience reduces error-driven incidents |
| `shift_duration_hrs` | Longer shifts → fatigue |
| `depth_m` | Deeper workings → harder ventilation, more stress |
| `temperature_c` | Heat stress affects judgment and stamina |
| `workforce_size` | More people, more exposure |
| `incidents_last_30_days` | Recency/momentum effect — problems compound |

**Target:** `incident_occurred` (binary) — incident rate ≈ 17%, realistic for
this kind of safety data (not 50/50, which would be a red flag if asked).

## 3. Approach

1. **EDA** — correlation heatmap + bucketed incident-rate charts to confirm
   the risk factors behave the way domain knowledge predicts (e.g. incident
   rate rises sharply below ~5 m³/min ventilation).
2. **Two models**, deliberately chosen to tell a complete story:
   - **Logistic Regression** — interpretable baseline, coefficients directly
     show direction/strength of each risk factor.
   - **Random Forest** — captures non-linear interactions (e.g. high methane
     *combined with* old machinery is worse than either alone), and gives
     feature importance.
3. **Class imbalance handled** with `class_weight="balanced"` — important to
   mention, since a naive model could hit 83% accuracy by just predicting
   "no incident" every time (base rate). Precision/Recall/ROC-AUC are the
   metrics that actually matter here, not raw accuracy.

## 4. Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 0.74 | 0.81 |
| Precision | 0.37 | 0.43 |
| Recall | 0.70 | 0.40 |
| F1 | 0.48 | 0.41 |
| ROC-AUC | 0.77 | 0.76 |

**Top risk factors (Random Forest):** methane level, ventilation rate,
machinery age, worker experience, depth.

## 5. The talking point that impresses interviewers

> "Logistic Regression has higher recall (0.70) — it catches more true
> incidents, which matters more in safety than precision, because a missed
> high-risk shift is far more costly than a false alarm. So even though
> Random Forest has higher accuracy, I'd deploy Logistic Regression, or tune
> the Random Forest's decision threshold down, because **in safety
> applications, recall on the positive class is the metric that matters, not
> overall accuracy.**"

This one sentence signals you understand ML *and* mining risk trade-offs —
that's the combination that gets you selected.

## 6. Files

- `generate_data.py` — creates the synthetic dataset from domain-driven rules
- `mine_shift_safety_data.csv` — the dataset
- `analysis.py` — EDA, both models, evaluation, all charts
- `01_correlation_heatmap.png`, `02_eda_key_factors.png`,
  `03_confusion_matrices.png`, `04_roc_curve.png`, `05_feature_importance.png`
- `results_summary.txt` — plain-text metrics summary

## 7. How to extend it (mention if asked "what would you do next?")

- Replace synthetic data with real incident logs (DGMS reports, company
  safety records) if available.
- Add a simple Streamlit dashboard so a supervisor can input today's shift
  parameters and get a live risk score.
- Try SHAP values instead of built-in feature importance for more rigorous
  explainability.
