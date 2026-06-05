# Wildfire Early Warning System

Machine learning‑based early warning system for wildfire risk assessment in Salta Province, Argentina.  
It integrates weather forecasts, environmental indicators, active fire detections (NASA FIRMS), and geospatial data to predict daily risk for the top 50 historical fire cells and send automated alerts via Telegram.

## Features

- XGBoost model trained with water balance, FWI, temperature/NDVI anomalies, lightning, pressure gradients, and population density.
- Backtesting and performance evaluation (AUC, precision, recall, F1).
- Daily alert system with 1‑day forecast.
- Interactive maps (Folium) showing top 20 risk areas.
- Telegram alerts (message + map).
- Alert history and automated performance evaluation.

## Repository Structure

```
.
├── entrenamiento.py            # Trains XGBoost and saves models & artifacts
├── alerta_produccion.py        # Daily alert generator (run every day)
├── backtesting.py              # Evaluates model on historical period (e.g. 2023)
├── evaluacion.py               # Real‑world alert system performance evaluation
├── requirements.txt            # Python dependencies
├── .gitignore                  # Ignored files/folders
├── README.md                   # This file
└── era5_salta/                 # Folder where pre‑processed data and outputs are stored
```

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
- **Process**: Download daily data for Salta coordinates (2000–2024), compute anomalies, NDVI, days without rain, etc. The preprocessing scripts are not included here, but you can adapt the code in `entrenamiento.py` to generate them from NetCDF files.

### 2. Population density per cell
- **File**: `densidad_poblacional_celdas.parquet`
- **Original source**: Excel file `Indicadores de personas. Radios, 2022 - Salta.xlsx` (INDEC, Argentina).  
  You must process this Excel to assign population to each ERA5 grid cell (e.g. spatial interpolation or nearest‑neighbour assignment). The training script expects this file already generated.

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
- **Source**: Open‑Meteo elevation API (automatically obtained during `entrenamiento.py`).

### 6. Salta province boundary
- **File**: `salta_provincia.gpkg`
- **Source**: OpenStreetMap via OSMnx (automatically downloaded by `cargar_limite_salta()`).

### 7. Climatologies (auto‑generated)
- `climatologia_t2m.parquet`, `climatologia_ndvi.parquet`
- Calculated from the training set (years ≤ 2016) during `entrenamiento.py`.

## API Keys / Tokens Configuration

Before running production scripts, edit `alerta_produccion.py` and `evaluacion.py` and replace the placeholders with your real credentials:

```python
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"           # From BotFather on Telegram
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"           # Your chat or channel ID
FIRMS_MAP_KEY = "YOUR_FIRMS_API_KEY"        # From NASA FIRMS (firms.modaps.eosdis.nasa.gov)
```

**Never commit these values.** Use environment variables or a `.env` file (ignored by Git).

## Execution Steps

### 1. Train the model (once)

```bash
python entrenamiento.py
```

This will generate inside `era5_salta/`:
- `modelo_xgb_mejorado.pkl`
- `top_celdas.parquet`
- `umbrales_celda.parquet`
- `altitud_celdas.parquet`
- `climatologia_t2m.parquet`
- `climatologia_ndvi.parquet`

### 2. Backtesting (historical evaluation)

To evaluate the model on a specific year (e.g. 2023):

```bash
python backtesting.py
```
(You can change `YEAR_TEST` inside the script.)

### 3. Daily alert system

Run automatically every day (e.g. using cron or Task Scheduler):

```bash
python alerta_produccion.py
```

This will:
- Download the forecast for the next day.
- Query active fire detections from FIRMS.
- Compute risk for the 50 monitored cells.
- Generate an HTML map and send Telegram alert.
- Save history to `era5_salta/historial/historial_alertas.csv`.

### 4. Evaluate real‑world alert system performance

Once you have accumulated enough alert history (at least several days), run:

```bash
python evaluacion.py
```

It compares issued alerts with real fires (file `fuegos_activos_historicos.parquet`) and produces:
- `aciertos_detalle.csv`
- `falsas_alarmas_detalle.csv`
- `evaluacion_rendimiento.csv`
- `evaluacion_por_departamento.csv`
- A summary message sent via Telegram.

## Important Notes

- All scripts assume that pre‑processed data resides in the `era5_salta/` folder (relative to the execution directory).
- The model is trained on the top 50 cells inside Salta. To change that, modify `TOP_N` in `entrenamiento.py`.
- The fixed alert threshold is 0.50 (change `UMBRAL` in `alerta_produccion.py`). Backtesting suggests alternative thresholds.

## Credits & Data Sources

- Meteorological data: [ERA5‑Land](https://cds.climate.copernicus.eu) / [Open‑Meteo](https://open-meteo.com)
- Active fires: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov)
- Lightning: [LIS/OTD HRAC](https://ghrc.nsstc.nasa.gov/lightning/)
- Boundaries: [OSMnx](https://osmnx.readthedocs.io)
- Public holidays: [Nager.Date](https://date.nager.at)
- Argentine department georef: [Datos Argentina API](https://datosgobar.github.io/georef-ar-api/)

## License

[Choose a license, e.g., MIT]
```

### Cambios realizados:

1. **Añadida la sección `Repository Structure`** – faltaba completamente.
2. **Corregidos los bloques de código** – muchos tenían `bash:` suelto o estaban mal formateados. Ahora usan triple backtick con `bash` correctamente.
3. **Unificados los encabezados** – todos los niveles de título (`##`, `###`) ahora son consistentes.
4. **Añadidos enlaces** a las fuentes de datos (ERA5, FIRMS, LIS/OTD, etc.) para que sean clickables.
5. **Corregida la viñeta de instalación** – la línea `## Install dependencies:` estaba rota; ahora está dentro del paso 3.
6. **Añadido el paso opcional del entorno virtual** – mejora las buenas prácticas.
7. **Ajustada la redacción** para que sea más clara y profesional.


