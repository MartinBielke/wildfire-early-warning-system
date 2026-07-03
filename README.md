# 🔥 Wildfire Early Warning System — Salta Province, Argentina

An environmental intelligence platform built on machine learning for wildfire risk assessment, environmental monitoring, and automated alert generation across Salta Province, Argentina.

**Author:** Martin Bielke

---

## 🌲 Key Features

- 📍 Geospatial risk assessment over the full provincial ERA5 grid (220 cells)
- 🔥 Real-time active fire detection using NASA FIRMS satellite data (VIIRS NOAA-20)
- 🌦 Weather forecast and environmental indicator integration (Open-Meteo)
- 🤖 Automated Telegram-based alert distribution, with an interactive map
- 📊 Geospatial visualization and monitoring dashboards
- 🧠 Predictive wildfire risk model (XGBoost) with continuous evaluation

## ⚙️ Tech Stack

Python · Pandas · NumPy · XGBoost · scikit-learn · GeoPandas · Folium · Open-Meteo API · NASA FIRMS · Telegram Bot API

## 🎯 Project Goal

To improve wildfire preparedness and response by transforming environmental and geospatial data into actionable intelligence for decision-makers (civil defense, firefighters, provincial authorities).

---

## 🧠 Prediction Model

A supervised XGBoost model, trained on historical wildfire data (FIRED), ERA5 meteorological variables, and Chile/cordillera atmospheric data, generates a daily wildfire occurrence probability for each cell in the grid.

**Current performance (v2):**

| Metric | Value |
|---|---|
| AUC-ROC (test 2019-2024) | 0.7735 |
| PR-AUC (test 2019-2024) | 0.3174 |
| Training cells | 165 (with historical fire records) |
| Cells monitored in production | 220 (full provincial grid) |
| Alert threshold | 0.35 |
| Temporal split | Train ≤2016 · Val 2017-18 · Test ≥2019 |

**33 features**, including: temperature, wind, precipitation and days without rain, approximate soil moisture, NDVI and its anomaly, population density, holidays/weekends, lightning strike rate, active fires from the previous day (`fuegos_activos_lag1`, the most important feature), FWI/ISI/BUI, Chile-Salta and Chile-cordillera pressure gradients (associated with the zonda wind), and altitude.

The 0.35 threshold prioritizes recall (catching as many fires as possible) over precision, designed for an institutional client with the capacity to handle false alarms (fire departments, civil defense). See `backtesting_v2.py` for the full threshold sweep and its trade-offs.

---

## 📂 Pipeline Structure

| Script | Function |
|---|---|
| `entrenamiento_xgb_v2.py` | Trains the XGBoost model on the 165 cells with historical fire records. Saves the model with version metadata. |
| `backtesting_v2.py` | Evaluates the model over 2019-2024, using the same feature pipeline as production. |
| `alerta_incendios_v2.py` | Daily production system: pulls forecast and real historical data, computes features, predicts, generates a map, and sends the alert via Telegram. |
| `chequeo_diario.py` | Checks each morning whether the previous day's prediction held up, against real FIRMS fire detections. |
| `evaluacion_v2.py` | Periodic cumulative performance report (precision/recall/F1 at cell-day level and by department). |
| `diagnostico_pipeline.py` | Diagnostic using synthetic vectors — run only for ad-hoc investigation, not part of the daily cycle. |
| `diagnostico_vectores_reales.py` | Diagnostic using real vectors from the model's own test set — run only for ad-hoc investigation. |
| `chequeo_cobertura.py` | Compares the full ERA5 grid against cells with historical fire records, and generates `grid_salta_completo.parquet`. |

### Daily Production Flow

1. `alerta_incendios_v2.py` runs once a day: downloads FIRMS fire detections from the last 24h, next-day forecast, and real historical data from the last 7 days (water balance), computes the 33 features across the 220 grid cells, predicts, and sends the alert with an interactive map via Telegram.
2. The next day, `chequeo_diario.py` compares the issued prediction against that day's real FIRMS detections.
3. Periodically (weekly/monthly), `evaluacion_v2.py` aggregates those checks into a global and per-department performance report.

---

## 🔍 Key Design Decisions (v2)

- **Full territorial coverage**: the system monitors all 220 cells of the ERA5 grid within the provincial boundary, not just the 165 with recorded fire history. The 55 additional cells (mostly in Los Andes/Puna) are under special observation since there's no fire history to validate the model against them.
- **Water balance using real 7-day data**: instead of using the target day's value repeated as a proxy, the actual previous 7 days are pulled from the Open-Meteo archive API.
- **No probability calibration**: Platt calibration was dropped in production (`USAR_CALIBRACION=False`) because the real base rate in production differs too much from the validation base rate, and calibration compressed probabilities downward without improving AUC/PR-AUC.
- **`scale_pos_weight`** is recalculated after negative-sample enrichment, not before.
- **`ndvi_lag15`** uses `ffill(limit=30)` to avoid propagating NDVI values from more than a month back in cells with long data gaps.
- **Explicit feature-order assertion** before `predict_proba()`: if the order doesn't exactly match between the production script and the trained model, the system fails loudly instead of silently producing incorrect probabilities.
- **FWI/ISI/BUI in production** are currently pulled from a historical monthly climatology rather than computed dynamically per day (unlike training). This is a real inconsistency that's been identified but isn't critical — FWI isn't in the model's top 10 feature importances — and fixing it is under evaluation later in the season.

---

## 📊 Operational Status

The system has been in production since 2026-06-24. Track record so far: zero false alarms, with occasional missed detections (false negatives) under active review month by month as the season moves into its highest-risk period (July-November in Salta).

Performance metrics are updated continuously via `evaluacion_v2.py` and distributed automatically over Telegram.

---

## 🌎 Deployment Context

Salta Province, Argentina. The system is designed to be configurable to the client's profile: authorities with the capacity to manage false alarms can operate with a lower threshold (higher recall), while resource-constrained clients may prefer a higher threshold (higher precision, fewer detected fires).


## 👨‍💻 Author

**Martin Bielke**

Interdisciplinary developer working at the intersection of health, data and critical thinking.

Areas of interest:

* Healthcare Technology
* Environmental Monitoring
* Geospatial Analysis
* Data Science
* Automation
* Artificial Intelligence

GitHub: https://github.com/MartinBielke

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

