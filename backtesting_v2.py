#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
backtesting_v2.py
=================
Backtesting del modelo XGBoost.

Cambio respecto a la versión anterior:
  - ndvi_lag15 usa ffill(limit=30), igual que entrenamiento_xgb_v2.py.
    La versión anterior usaba ffill() sin límite, lo que podía propagar
    valores de NDVI de más de un mes atrás en celdas con gaps largos de
    datos. Esto generaba una pequeña inconsistencia entre cómo se calculaba
    esta feature durante el entrenamiento y durante el backtesting —no
    afecta las métricas de forma significativa (los resultados ya
    reportados con AUC=0.7740 son prácticamente idénticos), pero mantiene
    el pipeline de features unificado entre ambos scripts, que es un
    invariante explícito documentado en entrenamiento_xgb_v2.py.

Incluye todas las features del modelo final (pressure_gradient_zonda,
altitud, etc.).
"""

import pandas as pd
import numpy as np
import pickle
import requests
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score

ERA5_DIR = Path("era5_salta")
YEAR_TEST = 2019  # Cambia a 2023 para un año específico


def obtener_feriados_argentina(anio):
    url = f"https://date.nager.at/api/v3/PublicHolidays/{anio}/AR"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return [h['date'] for h in response.json()]
    except Exception as e:
        print(f"⚠️ Error en feriados {anio}: {e}")
        return []


# ──────────────────────────────────────────────────────────────────
# Cargar modelo y top celdas
# ──────────────────────────────────────────────────────────────────
print("Cargando modelo y artefactos...")
with open(ERA5_DIR / "modelo_xgb_mejorado.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data['modelo']
features = model_data['features']

top_celdas = pd.read_parquet(ERA5_DIR / "top_celdas.parquet")
celdas_monitoreadas = top_celdas[['lat_era5', 'lon_era5']]

alt_celdas = pd.read_parquet(ERA5_DIR / "altitud_celdas.parquet")
alt_dict = alt_celdas.set_index(['lat_era5', 'lon_era5'])['altitud'].to_dict()

# ──────────────────────────────────────────────────────────────────
# Cargar datos base
# ──────────────────────────────────────────────────────────────────
print("Cargando datos base...")
fired = pd.read_parquet(ERA5_DIR / "fired_completo.parquet")
negat = pd.read_parquet(ERA5_DIR / "negativos.parquet")

pos = fired[['lat_era5', 'lon_era5', 't2m_max', 't2m_mean', 't2m_anomalia',
             'wind_max', 'wind_dir', 'precip', 'dias_sin_lluvia', 'sp_min',
             'sp_delta', 'swvl1', 'ndvi', 'ndvi_anomalia', 'date']].copy()
pos['fuego'] = 1
pos = pos.rename(columns={'date': 'fecha'})
pos = pos.merge(celdas_monitoreadas, on=['lat_era5', 'lon_era5'], how='inner')

neg = negat[['lat_era5', 'lon_era5', 't2m_max', 't2m_mean', 't2m_anomalia',
             'wind_max', 'wind_dir', 'precip', 'dias_sin_lluvia', 'sp_min',
             'sp_delta', 'swvl1', 'ndvi', 'ndvi_anomalia', 'fecha']].copy()
neg['fuego'] = 0
neg = neg.merge(celdas_monitoreadas, on=['lat_era5', 'lon_era5'], how='inner')

df = pd.concat([pos, neg], ignore_index=True)
df['fecha'] = pd.to_datetime(df['fecha'])
df.drop(columns=['sp_delta'], inplace=True, errors='ignore')
print(f"Dataset base (solo top celdas): {df.shape}")

# ──────────────────────────────────────────────────────────────────
# 1. Filtrar por fecha con contexto previo
# ──────────────────────────────────────────────────────────────────
# IMPORTANTE: se cargan 30 días previos al período de test para que las features
# con ventana temporal (ndvi_lag15, fuegos_activos_lag1) tengan contexto histórico
# real en los primeros días del período evaluado. Sin esto, el lag queda en 0 o
# en la media global para los primeros días, lo que distorsiona la evaluación dado
# que fuegos_activos_lag1 es la feature más importante del modelo (16.4%).
# Al final se filtra 'df_eval' con solo el período objetivo para calcular métricas.
DIAS_CONTEXTO = 30
if YEAR_TEST == 2019:
    fecha_inicio_contexto = pd.Timestamp('2018-12-01')
    fecha_inicio_eval = pd.Timestamp('2019-01-01')
    mask = df['fecha'] >= fecha_inicio_contexto
else:
    fecha_inicio_contexto = pd.Timestamp(f'{YEAR_TEST - 1}-12-01')
    fecha_inicio_eval = pd.Timestamp(f'{YEAR_TEST}-01-01')
    mask = (df['fecha'] >= fecha_inicio_contexto) & (df['fecha'].dt.year <= YEAR_TEST)

df = df[mask].copy()
print(f"Período cargado (con contexto): {df['fecha'].min().date()} a {df['fecha'].max().date()}")
print(f"Período de evaluación: {fecha_inicio_eval.date()} en adelante")

# ──────────────────────────────────────────────────────────────────
# 2. Cargar datos de Chile y cordillera
# ──────────────────────────────────────────────────────────────────
print("Cargando datos de Chile...")
chile_data = pd.read_parquet(ERA5_DIR / "datos_chile.parquet")
df = df.merge(chile_data, on='fecha', how='left')
df['pressure_gradient'] = (df['presion_chile_hpa'] * 100) - df['sp_min']
df['pressure_gradient'] = df['pressure_gradient'].fillna(0)
df['hr_min_chile'] = df['hr_min_chile'].fillna(df['hr_min_chile'].median())
df['wind_gust_max_chile'] = df['wind_gust_max_chile'].fillna(0)

print("Cargando datos de cordillera...")
cord_data = pd.read_parquet(ERA5_DIR / "presion_cordillera.parquet")
df = df.merge(cord_data, on='fecha', how='left')
df['presion_cordillera_hpa'] = df['presion_cordillera_hpa'].fillna(0)
df['pressure_gradient_zonda'] = (df['presion_chile_hpa'] - df['presion_cordillera_hpa']) * 100
df['pressure_gradient_zonda'] = df['pressure_gradient_zonda'].fillna(0)

# Altitud
df['altitud'] = df.apply(lambda row: alt_dict.get((row['lat_era5'], row['lon_era5']), 1000), axis=1)

# ──────────────────────────────────────────────────────────────────
# 3. Calcular todas las features
# ──────────────────────────────────────────────────────────────────
print("Calculando features...")
df['dia_año'] = df['fecha'].dt.dayofyear
df['sin_dia'] = np.sin(2 * np.pi * df['dia_año'] / 365.25)
df['cos_dia'] = np.cos(2 * np.pi * df['dia_año'] / 365.25)
df['mes'] = df['fecha'].dt.month

# Climatología t2m
clim_t2m = pd.read_parquet(ERA5_DIR / "climatologia_t2m.parquet")
df = df.merge(clim_t2m, on=['lat_era5', 'lon_era5', 'mes'], how='left')
df['t2m_anomalia_aprox'] = df['t2m_mean'] - df['t2m_clim']

# NDVI lag 15
# CAMBIO v2: ffill(limit=30) para ser consistente con entrenamiento_xgb_v2.py.
# La versión anterior usaba ffill() sin límite.
df = df.sort_values(['lat_era5', 'lon_era5', 'fecha'])
df['ndvi_lag15'] = df.groupby(['lat_era5', 'lon_era5'])['ndvi'].shift(15)
df['ndvi_lag15'] = (df
                    .groupby(['lat_era5', 'lon_era5'])['ndvi_lag15']
                    .ffill(limit=30))
df['ndvi_lag15'] = df['ndvi_lag15'].fillna(df['ndvi'].mean())

# Climatología NDVI
clim_ndvi = pd.read_parquet(ERA5_DIR / "climatologia_ndvi.parquet")
df = df.merge(clim_ndvi, on=['lat_era5', 'lon_era5', 'mes'], how='left')
df['ndvi_anomalia_aprox'] = df['ndvi_lag15'] - df['ndvi_clim']

# Balance hídrico
df['precip_7d'] = df.groupby(['lat_era5', 'lon_era5'])['precip'].transform(
    lambda x: x.rolling(7, min_periods=1).sum()
)
df['t2m_max_7d'] = df.groupby(['lat_era5', 'lon_era5'])['t2m_max'].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
df['t2m_mean_7d'] = df.groupby(['lat_era5', 'lon_era5'])['t2m_mean'].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
et_7d = 0.0023 * (df['t2m_mean_7d'] + 17.8) * (df['t2m_max_7d'] - df['t2m_mean_7d']).clip(lower=0) * 7
df['balance_hidrico_7d'] = df['precip_7d'] - et_7d
df['swvl1_aprox'] = np.clip(0.1 + 0.005 * df['balance_hidrico_7d'], 0.05, 0.35)

# Densidad poblacional
dens_pob = pd.read_parquet(ERA5_DIR / "densidad_poblacional_celdas.parquet")
df = df.merge(dens_pob, on=['lat_era5', 'lon_era5'], how='left')
df['densidad_poblacional'] = df['densidad_poblacional'].fillna(0)

# Rayos
rayos_df = pd.read_parquet(ERA5_DIR / "rayos_celdas.parquet")
df = df.merge(rayos_df, on=['lat_era5', 'lon_era5', 'dia_año'], how='left')
df['tasa_rayos'] = df['tasa_rayos'].fillna(0)

# Feriados
years = df['fecha'].dt.year.unique()
all_holidays = []
for year in years:
    all_holidays.extend(obtener_feriados_argentina(year))
holidays_set = set(all_holidays)
df['es_feriado'] = df['fecha'].dt.strftime('%Y-%m-%d').isin(holidays_set).astype(int)
df['es_finde_semana'] = (df['fecha'].dt.dayofweek >= 5).astype(int)

# Features avanzadas
df = df.sort_values(['lat_era5', 'lon_era5', 'fecha'])
df['t2m_max_3d_avg'] = df.groupby(['lat_era5', 'lon_era5'])['t2m_max'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)
df['precip_3d_sum'] = df.groupby(['lat_era5', 'lon_era5'])['precip'].transform(
    lambda x: x.rolling(3, min_periods=1).sum()
)
df['rayos_ndvi'] = df['tasa_rayos'] * df['ndvi_anomalia_aprox'].clip(lower=0)
df['temp_ndvi'] = df['t2m_anomalia_aprox'] * df['ndvi_anomalia_aprox'].clip(lower=0)
df['fire_danger'] = (df['t2m_max'] - 20).clip(lower=0) * (df['wind_max'] / 10) * (1 - df['swvl1_aprox'])

# Fuegos activos históricos con lag 1
fuegos_hist = pd.read_parquet(ERA5_DIR / "fuegos_activos_historicos.parquet")
fuegos_hist = fuegos_hist.merge(celdas_monitoreadas, on=['lat_era5', 'lon_era5'], how='inner')
df = df.merge(fuegos_hist, on=['lat_era5', 'lon_era5', 'fecha'], how='left')
df['fuegos_activos'] = df['fuegos_activos'].fillna(0).astype(int)
df = df.sort_values(['lat_era5', 'lon_era5', 'fecha'])
df['fuegos_activos_lag1'] = df.groupby(['lat_era5', 'lon_era5'])['fuegos_activos'].shift(1).fillna(0).astype(int)

# Humedad relativa
hr_hist = pd.read_parquet(ERA5_DIR / "hr_historico.parquet")
hr_hist['fecha'] = pd.to_datetime(hr_hist['fecha'])
df = df.merge(hr_hist, on=['lat_era5', 'lon_era5', 'fecha'], how='left')
df['hr_mean'] = df['hr_mean'].fillna(df['hr_mean'].median())
df['temp_hr'] = df['t2m_max'] * (100 - df['hr_mean']) / 100

# FWI
fwi_hist = pd.read_parquet(ERA5_DIR / "fwi_historico.parquet")
fwi_hist['fecha'] = pd.to_datetime(fwi_hist['fecha'])
df = df.merge(fwi_hist[['lat_era5', 'lon_era5', 'fecha', 'fwi', 'isi', 'bui']],
              on=['lat_era5', 'lon_era5', 'fecha'], how='left')
df['fwi'] = df['fwi'].fillna(0)
df['isi'] = df['isi'].fillna(0)
df['bui'] = df['bui'].fillna(0)

# ──────────────────────────────────────────────────────────────────
# 4. Recortar al período de evaluación real
# ──────────────────────────────────────────────────────────────────
# Las features se calcularon sobre el rango completo (con contexto), pero las
# métricas se evalúan solo sobre el período objetivo, sin los días de warmup.
df_eval = df[df['fecha'] >= fecha_inicio_eval].copy()
print(f"\nRegistros en período de evaluación: {len(df_eval)}")
print(f"  ({df_eval['fecha'].min().date()} a {df_eval['fecha'].max().date()})")

# NOTA DE COBERTURA: este backtesting evalúa solo las 165 celdas con incendio
# histórico registrado (top_celdas.parquet). Las 55 celdas adicionales del grid
# completo (grid_salta_completo.parquet, usado en alerta_incendios_v2.py) no tienen
# etiquetas fuego=0/1 en fired_completo.parquet, por lo que no pueden evaluarse
# aquí. Las métricas de abajo reflejan el comportamiento del modelo sobre zonas
# con historial conocido — pueden no representar exactamente el rendimiento sobre
# celdas sin historial previo (como la Puna).
print(f"\n⚠️ COBERTURA: backtesting sobre {celdas_monitoreadas.shape[0]} celdas con historial "
      f"de incendio. Las celdas sin historial (grid completo = 220) no se pueden evaluar aquí.")

missing = set(features) - set(df_eval.columns)
if missing:
    print(f"⚠️ Faltan columnas: {missing}")
    for col in missing:
        df_eval[col] = 0

X_test = df_eval[features]
y_test = df_eval['fuego']
print(f"Registros finales en test: {len(X_test)} | Positivos: {y_test.sum()}")

# ──────────────────────────────────────────────────────────────────
# 5. Predicción y evaluación
# ──────────────────────────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

# UMBRAL_FIJO es el umbral usado en producción por alerta_incendios_v2.py.
UMBRAL_FIJO = 0.35
UMBRAL_ALTERNATIVO = 0.50  # evaluado por completitud histórica

y_pred_fijo = (y_prob >= UMBRAL_FIJO).astype(int)
prec_fijo = precision_score(y_test, y_pred_fijo, zero_division=0)
rec_fijo = recall_score(y_test, y_pred_fijo, zero_division=0)
f1_fijo = f1_score(y_test, y_pred_fijo, zero_division=0)

y_pred_alt = (y_prob >= UMBRAL_ALTERNATIVO).astype(int)
prec_alt = precision_score(y_test, y_pred_alt, zero_division=0)
rec_alt = recall_score(y_test, y_pred_alt, zero_division=0)
f1_alt = f1_score(y_test, y_pred_alt, zero_division=0)

# Umbrales por celda
umbrales_celda = pd.read_parquet(ERA5_DIR / "umbrales_celda.parquet")
umbral_dict = {(row['lat_era5'], row['lon_era5']): row['umbral_celda'] for _, row in umbrales_celda.iterrows()}
df_eval['umbral'] = df_eval.apply(lambda row: umbral_dict.get((row['lat_era5'], row['lon_era5']), 0.35), axis=1)
y_pred_celda = (y_prob >= df_eval['umbral']).astype(int)
prec_celda = precision_score(y_test, y_pred_celda, zero_division=0)
rec_celda = recall_score(y_test, y_pred_celda, zero_division=0)
f1_celda = f1_score(y_test, y_pred_celda, zero_division=0)

print(f"\n{'='*50}")
print(f"AUC-ROC: {auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"Umbral producción ({UMBRAL_FIJO:.2f}) → Precisión: {prec_fijo:.3f}, Recall: {rec_fijo:.3f}, F1: {f1_fijo:.3f}  <- alerta_incendios_v2.py")
print(f"Umbral alternativo ({UMBRAL_ALTERNATIVO:.2f}) → Precisión: {prec_alt:.3f}, Recall: {rec_alt:.3f}, F1: {f1_alt:.3f}")
print(f"Umbrales por celda → Precisión: {prec_celda:.3f}, Recall: {rec_celda:.3f}, F1: {f1_celda:.3f}")
print(f"{'='*50}")

# ──────────────────────────────────────────────────────────────────
# Barrido de umbrales
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EVALUACIÓN CON DIFERENTES UMBRALES")
print("="*60)
umbrales = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
resultados_umbrales = []
for umbral in umbrales:
    y_pred = (y_prob >= umbral).astype(int)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    alertas = y_pred.sum()
    aciertos = (y_pred & y_test).sum()
    marca = " <- producción (alerta_incendios_v2.py)" if umbral == UMBRAL_FIJO else ""
    print(f"Umbral {umbral:.2f}: Precisión={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, Alertas={alertas}, Aciertos={aciertos}{marca}")
    resultados_umbrales.append({
        'umbral': umbral,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'alertas': alertas,
        'aciertos': aciertos
    })

df_umbrales = pd.DataFrame(resultados_umbrales)
print("\n" + "="*60)
print("RESUMEN:")
print(df_umbrales.to_string(index=False))

