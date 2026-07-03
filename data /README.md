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
- **Process**: Download daily data for Salta coordinates (2000–2024), compute anomalies, NDVI, days without rain, etc. The preprocessing scripts are not included here, but you can adapt the code inside the notebook to generate them from NetCDF files.

### 2. Population density per cell
- **File**: `densidad_poblacional_celdas.parquet`
- **Original source**: Excel file `Indicadores de personas. Radios, 2022 - Salta.xlsx` (INDEC, Argentina).  
  You must process this Excel to assign population to each ERA5 grid cell (e.g. spatial interpolation or nearest‑neighbour assignment). The notebook expects this file already generated.

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
- **Source**: Open‑Meteo elevation API (automatically obtained during the training section of the notebook).

### 6. Salta province boundary
- **File**: `salta_provincia.gpkg`
- **Source**: OpenStreetMap via OSMnx (automatically downloaded by `cargar_limite_salta()` function inside the notebook).

### 7. Climatologies (auto‑generated)
- `climatologia_t2m.parquet`, `climatologia_ndvi.parquet`
- Calculated from the training set (years ≤ 2016) during the training cells.

## API Keys / Tokens Configuration

Before running the alert parts of the notebook, open `Wildfire_decision_support_system.ipynb` and replace the placeholders in the configuration cell with your real credentials:

```python
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"           # From BotFather on Telegram
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"           # Your chat or channel ID
FIRMS_MAP_KEY = "YOUR_FIRMS_API_KEY"        # From NASA FIRMS (firms.modaps.eosdis.nasa.gov)
```

**Never commit these values.** Use environment variables or a `.env` file (ignored by Git).

## Execution Steps

All code is contained in a single Jupyter notebook: `Wildfire_decision_support_system.ipynb`.  
Open it with Jupyter Lab, Jupyter Notebook, or VS Code and execute the cells **in order**.

The notebook is divided into well‑marked sections:

1. **Configuration** – set paths, tokens, thresholds.
2. **Data loading & preprocessing** – expects pre‑processed `.parquet` files in `era5_salta/`.
3. **Training** – runs XGBoost and saves model artifacts.
4. **Backtesting** – evaluates model on a chosen historical year (e.g. 2023).
5. **Daily alert system** – generates next‑day risk alerts (run this section daily).
6. **Evaluation** – compares issued alerts with real fires.

### Step‑by‑step

1. **Launch the notebook**:
```bash
   jupyter notebook Wildfire_decision_support_system.ipynb
```
   or
```bash
   jupyter lab Wildfire_decision_support_system.ipynb
```

2. **Run cells sequentially** – execute from top to bottom.

   - **Training cells** (one‑time execution) → generates models and artifacts inside `era5_salta/`.
   - **Backtesting cells** (optional) → evaluate historical performance (change `YEAR_TEST` in the corresponding cell).
   - **Alert cells** – run daily to get next‑day risk maps and Telegram notifications.  
     *Important*: Each time you run these cells, they will download forecasts, query FIRMS, and send Telegram messages.
   - **Evaluation cells** – run after several days of alerts to measure real‑world performance.

3. **Scheduling daily alerts** (recommended for production):  
   Export the alert section of the notebook to a Python script:
```bash
   jupyter nbconvert --to script --output alerta_diaria Wildfire_decision_support_system.ipynb
```
   Then edit `alerta_diaria.py` to keep only the alert‑related code (or use the `--template` option). Schedule the script to run once per day (e.g., at 08:00 local time) using cron (Linux/Mac) or Task Scheduler (Windows).

## Important Notes

- All code assumes that pre‑processed data resides in the `era5_salta/` folder (relative to the notebook location).
- The notebook is designed to be run **linearly**. Some cells may take several minutes (training, backtesting).
- Before running alert cells, ensure you have set your `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, and `FIRMS_MAP_KEY` in the configuration cell.
- The alert system uses a fixed threshold of 0.35, prioritizing recall over precision. Change the `UMBRAL` variable inside the notebook if needed — see the threshold sweep in the Production section for trade-offs at other values.
- The model is trained on all 165 cells within Salta that have historical fire records (`TOP_N = None`). To restrict training to a subset, set `TOP_N` in the training section.

## Credits & Data Sources

- Meteorological data: [ERA5‑Land](https://cds.climate.copernicus.eu) / [Open‑Meteo](https://open-meteo.com)
- Active fires: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov)
- Lightning: [LIS/OTD HRAC](https://ghrc.nsstc.nasa.gov/lightning/)
- Boundaries: [OSMnx](https://osmnx.readthedocs.io)
- Public holidays: [Nager.Date](https://date.nager.at)
- Argentine department georef: [Datos Argentina API](https://datosgobar.github.io/georef-ar-api/)

## License

[Choose a license, e.g., MIT]
