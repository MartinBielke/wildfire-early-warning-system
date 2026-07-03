# Wildfire Early Warning System

Machine learning‑based early warning system for wildfire risk assessment in Salta Province, Argentina.  
It integrates weather forecasts, environmental indicators, active fire detections (NASA FIRMS), and geospatial data to predict daily risk across the full provincial grid (220 cells, including 165 with historical fire records) and send automated alerts via Telegram.

## Features

- XGBoost model trained with water balance, FWI, temperature/NDVI anomalies, lightning, pressure gradients, and population density.
- Backtesting and performance evaluation (AUC, precision, recall, F1).
- Daily alert system with 1‑day forecast.
- Interactive maps (Folium) showing top 20 risk areas.
- Telegram alerts (message + map).
- Alert history and automated performance evaluation.

## Requirements

- Python 3.8+
- Internet connection for:
  - Open‑Meteo forecasts
  - ERA5 historical data (pre‑downloaded)
  - NASA FIRMS API
  - Argentine georef API
  - Public holidays API (date.nager.at)

## Installation

1. Clone the repository:
```bash
   git clone https://github.com/MartinBielke/wildfire-early-warning-system.git
   cd wildfire-early-warning-system
```

2. (Optional) Create and activate a virtual environment:
```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

## Required Data – How to Obtain It

The system expects several pre‑processed `.parquet` files inside the `era5_salta/` folder.  
**These files are NOT included in the repository** – they must be generated (or downloaded) from the sources described below.

### 1. Historical meteorological & fire data (ERA5)
- **Files**: `fired_completo.parquet`, `negativos.parquet`, `fwi_historico.parquet`, `hr_historico.parquet`, `presion_cordillera.parquet`, `datos_chile.parquet`
- **Source**: [ERA5‑Land](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview) (temperature, wind, precipitation, humidity, pressure) and [FWI](https://cds.climate.copernicus.eu/cdsapp#!/dataset/cems-fire-historical?tab=overview)
- **Process**: Download daily data for Salta coordinates (2000–2024), compute anomalies, NDVI, days without rain, etc. The preprocessing scripts are not included here, but you can adapt the code inside `entrenamiento_xgb_v2.py` to generate them from NetCDF files.

### 2. Population density per cell
- **File**: `densidad_poblacional_celdas.parquet`
- **Original source**: Excel file `Indicadores de personas. Radios, 2022 - Salta.xlsx` (INDEC, Argentina).  
  You must process this Excel to assign population to each ERA5 grid cell (e.g. spatial interpolation or nearest‑neighbour assignment). The scripts expect this file already generated.

### 3. Lightning (climatological flash rate)
- **File**: `rayos_celdas.parquet`
- **Source**: [LIS/OTD HRAC](https://ghrc.nsstc.nasa.gov/lightning/data/data_lis-otd-hrac.html) (NASA)  
  Contains flash rate per cell and day‑of‑year. Process the NetCDF `LISOTD_HRAC_V2.3.2015.nc` to produce this file.

### 4. Historical active fires (for lag‑1 feature)
- **File**: `fuegos_activos_historicos.parquet`
- **Source**: NASA FIRMS (historical archive)  
  Used to create the feature `fuegos_activos_lag1` (active fires on the previous day). Download from [FIRMS](https://firms.modaps.eosdis.nasa.gov/download/).

### 5. Cell altitude
- **File**: `altitud_celdas.parquet`
- **Source**: Open‑Meteo elevation API (automatically obtained by `entrenamiento_xgb_v2.py` when run for the first time).

### 6. Salta province boundary
- **File**: `salta_provincia.gpkg`
- **Source**: OpenStreetMap via OSMnx (automatically downloaded by the `cargar_limite_salta()` function, shared across scripts).

### 7. Climatologies (auto‑generated)
- `climatologia_t2m.parquet`, `climatologia_ndvi.parquet`
- Calculated from the training set (years ≤ 2016) by `entrenamiento_xgb_v2.py`.

## API Keys / Tokens Configuration

Before running `alerta_incendios_v2.py` or `chequeo_diario.py`, open the script and replace the placeholders in the configuration section with your real credentials:

```python
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"           # From BotFather on Telegram
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"           # Your chat or channel ID
FIRMS_MAP_KEY = "YOUR_FIRMS_API_KEY"        # From NASA FIRMS (firms.modaps.eosdis.nasa.gov)
```

**Never commit these values.** Use environment variables or a `.env` file (ignored by Git).

## Execution Steps

The pipeline is organized as a set of standalone Python scripts, meant to be run in order:

1. **`entrenamiento_xgb_v2.py`** – trains the model (one‑time execution, or after adding new historical data). Generates model and artifacts inside `era5_salta/`.
2. **`backtesting_v2.py`** – evaluates the model on a chosen historical period (optional; change `YEAR_TEST` at the top of the script).
3. **`chequeo_cobertura.py`** – regenerates `grid_salta_completo.parquet` (only needed the first time, or after retraining).
4. **`alerta_incendios_v2.py`** – generates the next‑day risk alert and sends it via Telegram. **Run this one daily.**
5. **`chequeo_diario.py`** – checks the previous day's prediction against real FIRMS detections. Run daily, after `alerta_incendios_v2.py`.
6. **`evaluacion_v2.py`** – periodic cumulative performance report. Run weekly or monthly.
7. **`diagnostico_pipeline.py`** / **`diagnostico_vectores_reales.py`** – ad‑hoc diagnostics, run only when investigating unexpected model behavior. Not part of the daily cycle.

### Step‑by‑step

```bash
# One-time setup
python entrenamiento_xgb_v2.py
python chequeo_cobertura.py

# Optional: evaluate on historical data
python backtesting_v2.py

# Daily cycle
python alerta_incendios_v2.py
python chequeo_diario.py

# Periodic (weekly/monthly)
python evaluacion_v2.py
```

### Scheduling daily alerts (recommended for production)

Schedule `alerta_incendios_v2.py` followed by `chequeo_diario.py` to run once per day (e.g., at 08:00 local time) using cron (Linux/Mac) or Task Scheduler (Windows). Example crontab entry:

```bash
0 8 * * * cd /path/to/wildfire-early-warning-system && python alerta_incendios_v2.py && python chequeo_diario.py
```

## Important Notes

- All scripts assume that pre‑processed data resides in the `era5_salta/` folder (relative to each script's location).
- Some scripts may take several minutes to run (training, backtesting).
- Before running `alerta_incendios_v2.py` or `chequeo_diario.py`, ensure you have set your `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, and `FIRMS_MAP_KEY`.
- The alert system uses a fixed threshold of 0.35, prioritizing recall over precision. Change the `UMBRAL` variable in `alerta_incendios_v2.py` if needed — see the threshold sweep in the Production section of the main README for trade-offs at other values.
- The model is trained on all 165 cells within Salta that have historical fire records (`TOP_N = None` in `entrenamiento_xgb_v2.py`). To restrict training to a subset, set `TOP_N` accordingly.

## Credits & Data Sources

- Meteorological data: [ERA5‑Land](https://cds.climate.copernicus.eu) / [Open‑Meteo](https://open-meteo.com)
- Active fires: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov)
- Lightning: [LIS/OTD HRAC](https://ghrc.nsstc.nasa.gov/lightning/)
- Boundaries: [OSMnx](https://osmnx.readthedocs.io)
- Public holidays: [Nager.Date](https://date.nager.at)
- Argentine department georef: [Datos Argentina API](https://datosgobar.github.io/georef-ar-api/)

## License

[Choose a license, e.g., MIT]
