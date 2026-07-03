#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
entrenamiento_xgb_v2.py
=======================

  1. scale_pos_weight se recalcula DESPUÉS del enriquecimiento de negativos.
  2. ndvi_lag15 usa ffill(limit=30) para evitar propagar valores de meses anteriores.
  3. El calibrador NO se entrena ni se guarda (USAR_CALIBRACION=False en producción
     y la tasa base en producción difiere demasiado de la de val para que sea útil).
  4. El pickle incluye metadatos de entrenamiento: fecha, rango temporal, n_celdas,
     versión, para que alerta_incendios.py pueda verificar coherencia.
  5. Comentarios de advertencia actualizados donde corresponde.

Invariantes que se mantienen:
  - TOP_N = None  →  se usan TODAS las celdas con historial de incendio (165).
  - División temporal estricta: train ≤ 2016 | val 2017-18 | test ≥ 2019.
  - Climatologías calculadas solo con train (sin data leakage).
  - El pipeline de features es idéntico al de alerta_incendios.py (mismo orden,
    mismos nombres, mismas unidades).  Cualquier cambio aquí debe replicarse allá.
"""

import pandas as pd
import numpy as np
import xarray as xr
import requests
import geopandas as gpd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    average_precision_score, f1_score
)
from sklearn.neighbors import BallTree
import xgboost as xgb
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

# ========================= CONFIGURACIÓN =========================
ERA5_DIR = Path("era5_salta")
RANDOM_STATE = 42
TOP_N = None          # None → todas las celdas con historial de incendio
ENRIQUECER = True
CONDICIONES_EXTREMAS = {
    't2m_max':        35,
    'dias_sin_lluvia': 10,
    'wind_max':        7,
    'swvl1_aprox':     0.10,
}
FACTOR_DUPLICACION = 3

# Umbral provisional usado en el backtesting interno.
# alerta_incendios.py tiene su propio UMBRAL (ajustarlo allá independientemente).
UMBRAL_UNICO_PROVISIONAL = 0.35

VERSION_MODELO = "v2"
# =================================================================


# ──────────────────────────────────────────────────────────────────
# Funciones auxiliares
# ──────────────────────────────────────────────────────────────────

def cargar_limite_salta():
    limite_path = ERA5_DIR / "salta_provincia.gpkg"
    if not limite_path.exists():
        print("Descargando límite de Salta...")
        import osmnx as ox
        salta = ox.geocode_to_gdf("Provincia de Salta, Argentina")
        salta = salta.to_crs("EPSG:4326")
        salta.to_file(limite_path, driver='GPKG')
    return gpd.read_file(limite_path)


def obtener_departamento(lat, lon, cache={}):
    clave = (round(lat, 2), round(lon, 2))
    if clave in cache:
        return cache[clave]
    url = f"https://apis.datos.gob.ar/georef/api/ubicacion?lat={lat}&lon={lon}"
    try:
        response = requests.get(url, timeout=3)
        data = response.json()
        if 'ubicacion' in data and 'departamento' in data['ubicacion']:
            depto = data['ubicacion']['departamento']['nombre']
            cache[clave] = depto
            return depto
    except Exception:
        pass
    cache[clave] = "Desconocido"
    return "Desconocido"


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
# 1. Cargar datos base y top celdas
# ──────────────────────────────────────────────────────────────────
print("Cargando datos base...")
fired = pd.read_parquet(ERA5_DIR / "fired_completo.parquet")
negat = pd.read_parquet(ERA5_DIR / "negativos.parquet")
limite_salta = cargar_limite_salta()

gdf_fired = gpd.GeoDataFrame(
    fired,
    geometry=gpd.points_from_xy(fired['lon_era5'], fired['lat_era5']),
    crs="EPSG:4326"
)
dentro = gpd.sjoin(gdf_fired, limite_salta[['geometry']], how='inner', predicate='within')
fired_salta = fired.loc[dentro.index].copy()
print(f"Incendios totales: {len(fired)} | Dentro de Salta: {len(fired_salta)}")

celdas_unicas = fired_salta[['lat_era5', 'lon_era5']].drop_duplicates()
print(f"Celdas ERA5 únicas con incendios en Salta: {celdas_unicas.shape}")

frecuencia = (fired_salta
              .groupby(['lat_era5', 'lon_era5'])
              .size()
              .reset_index(name='frecuencia'))

top_celdas = frecuencia.sort_values('frecuencia', ascending=False).head(TOP_N).copy()
print(f"Total de celdas en el dataset (TOP_N={TOP_N}): {len(top_celdas)}")
print(top_celdas.head(10))

# Departamentos (con cache compartido con chequeo_cobertura.py)
print("Obteniendo departamentos...")
departamentos_path = ERA5_DIR / "departamentos_celdas.parquet"
if departamentos_path.exists():
    depto_cache = pd.read_parquet(departamentos_path)
    print(f"  Cache encontrado: {len(depto_cache)} celdas")
else:
    depto_cache = pd.DataFrame(columns=['lat_era5', 'lon_era5', 'departamento'])

celdas_con_depto_cache = top_celdas[['lat_era5', 'lon_era5']].merge(
    depto_cache, on=['lat_era5', 'lon_era5'], how='left'
)
celdas_depto_faltantes = celdas_con_depto_cache[
    celdas_con_depto_cache['departamento'].isna()
][['lat_era5', 'lon_era5']]
print(f"  Celdas nuevas a consultar: {len(celdas_depto_faltantes)}")

nuevos_deptos = []
for _, row in celdas_depto_faltantes.iterrows():
    depto = obtener_departamento(row['lat_era5'], row['lon_era5'])
    nuevos_deptos.append({'lat_era5': row['lat_era5'],
                          'lon_era5': row['lon_era5'],
                          'departamento': depto})
    time.sleep(0.2)

if nuevos_deptos:
    depto_cache = pd.concat([depto_cache, pd.DataFrame(nuevos_deptos)], ignore_index=True)
    depto_cache = depto_cache.drop_duplicates(subset=['lat_era5', 'lon_era5'], keep='last')
    depto_cache.to_parquet(departamentos_path, index=False)

top_celdas = top_celdas.merge(depto_cache, on=['lat_era5', 'lon_era5'], how='left')
top_celdas.to_parquet(ERA5_DIR / "top_celdas.parquet", index=False)
print(f"✅ top_celdas.parquet guardado ({len(top_celdas)} celdas)")

# Umbrales por celda (umbral único provisional)
top_celdas['umbral_celda'] = UMBRAL_UNICO_PROVISIONAL
top_celdas[['lat_era5', 'lon_era5', 'umbral_celda']].to_parquet(
    ERA5_DIR / "umbrales_celda.parquet", index=False
)
print(f"⚠️ Umbral único provisional ({UMBRAL_UNICO_PROVISIONAL}) aplicado a todas las celdas.")


# ──────────────────────────────────────────────────────────────────
# 2. Altitud de cada celda (con cache)
# ──────────────────────────────────────────────────────────────────
print("Obteniendo altitud de cada celda...")

def obtener_altitud(lat, lon):
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()['elevation'][0]

altitud_path = ERA5_DIR / "altitud_celdas.parquet"
if altitud_path.exists():
    alt_cache = pd.read_parquet(altitud_path)
    print(f"  Cache encontrado: {len(alt_cache)} celdas")
else:
    alt_cache = pd.DataFrame(columns=['lat_era5', 'lon_era5', 'altitud'])

celdas_con_cache = top_celdas[['lat_era5', 'lon_era5']].merge(
    alt_cache, on=['lat_era5', 'lon_era5'], how='left'
)
celdas_faltantes = celdas_con_cache[
    celdas_con_cache['altitud'].isna()
][['lat_era5', 'lon_era5']]
print(f"  Celdas sin altitud en cache: {len(celdas_faltantes)}")

nuevas_altitudes = []
for _, row in celdas_faltantes.iterrows():
    alt = obtener_altitud(row['lat_era5'], row['lon_era5'])
    nuevas_altitudes.append({'lat_era5': row['lat_era5'],
                             'lon_era5': row['lon_era5'],
                             'altitud': alt})
    time.sleep(0.1)

if nuevas_altitudes:
    alt_cache = pd.concat([alt_cache, pd.DataFrame(nuevas_altitudes)], ignore_index=True)
    alt_cache = alt_cache.drop_duplicates(subset=['lat_era5', 'lon_era5'], keep='last')
    alt_cache.to_parquet(altitud_path, index=False)

top_celdas = top_celdas.merge(alt_cache, on=['lat_era5', 'lon_era5'], how='left')
print(f"✅ altitud_celdas.parquet guardado ({len(alt_cache)} celdas)")


# ──────────────────────────────────────────────────────────────────
# 3. Construir dataset (solo top celdas)
# ──────────────────────────────────────────────────────────────────
pos = fired_salta[['lat_era5', 'lon_era5', 't2m_max', 't2m_mean', 't2m_anomalia',
                   'wind_max', 'wind_dir', 'precip', 'dias_sin_lluvia', 'sp_min',
                   'sp_delta', 'swvl1', 'ndvi', 'ndvi_anomalia', 'date']].copy()
pos['fuego'] = 1
pos = pos.rename(columns={'date': 'fecha'})
pos = pos.merge(top_celdas[['lat_era5', 'lon_era5']], on=['lat_era5', 'lon_era5'], how='inner')

neg = negat[['lat_era5', 'lon_era5', 't2m_max', 't2m_mean', 't2m_anomalia',
             'wind_max', 'wind_dir', 'precip', 'dias_sin_lluvia', 'sp_min',
             'sp_delta', 'swvl1', 'ndvi', 'ndvi_anomalia', 'fecha']].copy()
neg['fuego'] = 0
neg = neg.merge(top_celdas[['lat_era5', 'lon_era5']], on=['lat_era5', 'lon_era5'], how='inner')

df = pd.concat([pos, neg], ignore_index=True)
df['fecha'] = pd.to_datetime(df['fecha'])
df.drop(columns=['sp_delta'], inplace=True, errors='ignore')
print(f"Dataset base (solo top celdas): {df.shape}")


# ──────────────────────────────────────────────────────────────────
# 4. Datos de Chile y cordillera
# ──────────────────────────────────────────────────────────────────
print("Cargando datos de Chile y cordillera...")
chile_data = pd.read_parquet(ERA5_DIR / "datos_chile.parquet")
chile_data['fecha'] = pd.to_datetime(chile_data['fecha'])
df = df.merge(chile_data, on='fecha', how='left')

cord_data = pd.read_parquet(ERA5_DIR / "presion_cordillera.parquet")
cord_data['fecha'] = pd.to_datetime(cord_data['fecha'])
df = df.merge(cord_data, on='fecha', how='left')

df['pressure_gradient'] = (df['presion_chile_hpa'] * 100) - df['sp_min']
df['pressure_gradient'] = df['pressure_gradient'].fillna(0)
df['pressure_gradient_zonda'] = (df['presion_chile_hpa'] - df['presion_cordillera_hpa']) * 100
df['pressure_gradient_zonda'] = df['pressure_gradient_zonda'].fillna(0)
df['hr_min_chile'] = df['hr_min_chile'].fillna(df['hr_min_chile'].median())
df['wind_gust_max_chile'] = df['wind_gust_max_chile'].fillna(0)

alt_celdas = pd.read_parquet(ERA5_DIR / "altitud_celdas.parquet")
df = df.merge(alt_celdas, on=['lat_era5', 'lon_era5'], how='left')
df['altitud'] = df['altitud'].fillna(df['altitud'].median())
print(f"Años en df: {df['fecha'].dt.year.min()} a {df['fecha'].dt.year.max()}")


# ──────────────────────────────────────────────────────────────────
# 5. Columnas temporales básicas
# ──────────────────────────────────────────────────────────────────
df['dia_año'] = df['fecha'].dt.dayofyear
df['sin_dia'] = np.sin(2 * np.pi * df['dia_año'] / 365.25)
df['cos_dia'] = np.cos(2 * np.pi * df['dia_año'] / 365.25)
df['mes']     = df['fecha'].dt.month


# ──────────────────────────────────────────────────────────────────
# 6. División temporal
# ──────────────────────────────────────────────────────────────────
print("Dividiendo en train/val/test...")
df = df.sort_values(['lat_era5', 'lon_era5', 'fecha'])
train_df = df[df['fecha'].dt.year <= 2016].copy()
val_df   = df[(df['fecha'].dt.year >= 2017) & (df['fecha'].dt.year <= 2018)].copy()
test_df  = df[df['fecha'].dt.year >= 2019].copy()
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


# ──────────────────────────────────────────────────────────────────
# 7. Climatologías (solo con train → sin data leakage)
# ──────────────────────────────────────────────────────────────────
print("Calculando climatologías a partir de train...")
clim_t2m = train_df.groupby(['lat_era5', 'lon_era5', 'mes'])['t2m_mean'].mean().reset_index()
clim_t2m.rename(columns={'t2m_mean': 't2m_clim'}, inplace=True)

clim_ndvi = train_df.groupby(['lat_era5', 'lon_era5', 'mes'])['ndvi'].mean().reset_index()
clim_ndvi.rename(columns={'ndvi': 'ndvi_clim'}, inplace=True)

clim_t2m.to_parquet(ERA5_DIR / "climatologia_t2m.parquet", index=False)
clim_ndvi.to_parquet(ERA5_DIR / "climatologia_ndvi.parquet", index=False)
print("✅ Climatologías guardadas")


# ──────────────────────────────────────────────────────────────────
# 8. Función de features (compartida con backtesting y producción)
# ──────────────────────────────────────────────────────────────────
def apply_features(df_chunk, clim_t2m, clim_ndvi):
    df_chunk = df_chunk.copy()

    # Climatología t2m
    df_chunk = df_chunk.merge(clim_t2m, on=['lat_era5', 'lon_era5', 'mes'], how='left')
    df_chunk['t2m_anomalia_aprox'] = df_chunk['t2m_mean'] - df_chunk['t2m_clim']

    # NDVI lag 15
    # CAMBIO v2: limit=30 evita propagar valores de más de un mes atrás.
    df_chunk = df_chunk.sort_values(['lat_era5', 'lon_era5', 'fecha'])
    df_chunk['ndvi_lag15'] = df_chunk.groupby(['lat_era5', 'lon_era5'])['ndvi'].shift(15)
    df_chunk['ndvi_lag15'] = (df_chunk
                              .groupby(['lat_era5', 'lon_era5'])['ndvi_lag15']
                              .ffill(limit=30))
    df_chunk['ndvi_lag15'] = df_chunk['ndvi_lag15'].fillna(df_chunk['ndvi'].mean())

    # Climatología NDVI
    df_chunk = df_chunk.merge(clim_ndvi, on=['lat_era5', 'lon_era5', 'mes'], how='left')
    df_chunk['ndvi_anomalia_aprox'] = df_chunk['ndvi_lag15'] - df_chunk['ndvi_clim']

    # Balance hídrico (ventanas reales de 7 días)
    df_chunk['precip_7d'] = df_chunk.groupby(['lat_era5', 'lon_era5'])['precip'].transform(
        lambda x: x.rolling(7, min_periods=1).sum()
    )
    df_chunk['t2m_max_7d'] = df_chunk.groupby(['lat_era5', 'lon_era5'])['t2m_max'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df_chunk['t2m_mean_7d'] = df_chunk.groupby(['lat_era5', 'lon_era5'])['t2m_mean'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    et_7d = (0.0023
             * (df_chunk['t2m_mean_7d'] + 17.8)
             * (df_chunk['t2m_max_7d'] - df_chunk['t2m_mean_7d']).clip(lower=0)
             * 7)
    df_chunk['balance_hidrico_7d'] = df_chunk['precip_7d'] - et_7d
    df_chunk['swvl1_aprox'] = np.clip(0.1 + 0.005 * df_chunk['balance_hidrico_7d'], 0.05, 0.35)

    return df_chunk


# ──────────────────────────────────────────────────────────────────
# 9. Aplicar features a train / val / test
# ──────────────────────────────────────────────────────────────────
print("Aplicando features a train/val/test...")
train_df = apply_features(train_df, clim_t2m, clim_ndvi)
val_df   = apply_features(val_df,   clim_t2m, clim_ndvi)
test_df  = apply_features(test_df,  clim_t2m, clim_ndvi)


# ──────────────────────────────────────────────────────────────────
# 10. Tablas estáticas
# ──────────────────────────────────────────────────────────────────
print("Integrando tablas estáticas...")

# Densidad poblacional
dens_pob = pd.read_parquet(ERA5_DIR / "densidad_poblacional_celdas.parquet")
for dset in [train_df, val_df, test_df]:
    dset_ref = dset  # in-place merge requires reassignment below
for name, dset in [('train', train_df), ('val', val_df), ('test', test_df)]:
    merged = dset.merge(dens_pob, on=['lat_era5', 'lon_era5'], how='left')
    merged['densidad_poblacional'] = merged['densidad_poblacional'].fillna(0)
    if name == 'train':   train_df = merged
    elif name == 'val':   val_df   = merged
    else:                 test_df  = merged

# Rayos
rayos_df = pd.read_parquet(ERA5_DIR / "rayos_celdas.parquet")
for name, dset in [('train', train_df), ('val', val_df), ('test', test_df)]:
    merged = dset.merge(rayos_df, on=['lat_era5', 'lon_era5', 'dia_año'], how='left')
    merged['tasa_rayos'] = merged['tasa_rayos'].fillna(0)
    if name == 'train':   train_df = merged
    elif name == 'val':   val_df   = merged
    else:                 test_df  = merged

# Feriados (cache)
feriados_cache = ERA5_DIR / "feriados.parquet"
if not feriados_cache.exists():
    all_holidays = []
    for year in range(2000, 2025):
        all_holidays.extend(obtener_feriados_argentina(year))
    feriados_df = pd.DataFrame({'fecha': all_holidays})
    feriados_df['fecha'] = pd.to_datetime(feriados_df['fecha'])
    feriados_df.to_parquet(feriados_cache, index=False)
feriados_df = pd.read_parquet(feriados_cache)
feriados_set = set(feriados_df['fecha'].dt.strftime('%Y-%m-%d'))

for dset in [train_df, val_df, test_df]:
    dset['es_feriado']      = dset['fecha'].dt.strftime('%Y-%m-%d').isin(feriados_set).astype(int)
    dset['es_finde_semana'] = (dset['fecha'].dt.dayofweek >= 5).astype(int)

# Humedad relativa
hr_hist = pd.read_parquet(ERA5_DIR / "hr_historico.parquet")
hr_hist['fecha'] = pd.to_datetime(hr_hist['fecha'])
for name, dset in [('train', train_df), ('val', val_df), ('test', test_df)]:
    merged = dset.merge(hr_hist, on=['lat_era5', 'lon_era5', 'fecha'], how='left')
    merged['hr_mean'] = merged['hr_mean'].fillna(merged['hr_mean'].median())
    merged['temp_hr'] = merged['t2m_max'] * (100 - merged['hr_mean']) / 100
    if name == 'train':   train_df = merged
    elif name == 'val':   val_df   = merged
    else:                 test_df  = merged

# FWI
fwi_hist = pd.read_parquet(ERA5_DIR / "fwi_historico.parquet")
fwi_hist['fecha'] = pd.to_datetime(fwi_hist['fecha'])
for name, dset in [('train', train_df), ('val', val_df), ('test', test_df)]:
    merged = dset.merge(fwi_hist[['lat_era5', 'lon_era5', 'fecha', 'fwi', 'isi', 'bui']],
                        on=['lat_era5', 'lon_era5', 'fecha'], how='left')
    for col in ['fwi', 'isi', 'bui']:
        merged[col] = merged[col].fillna(0)
    if name == 'train':   train_df = merged
    elif name == 'val':   val_df   = merged
    else:                 test_df  = merged


# ──────────────────────────────────────────────────────────────────
# 11. Enriquecimiento de negativos (solo train)
# ──────────────────────────────────────────────────────────────────
if ENRIQUECER:
    print("Enriqueciendo muestras negativas en train...")
    mask_neg = train_df['fuego'] == 0
    mask_extremo = (
        (train_df.loc[mask_neg, 't2m_max']        > CONDICIONES_EXTREMAS['t2m_max'])
        & (train_df.loc[mask_neg, 'dias_sin_lluvia'] > CONDICIONES_EXTREMAS['dias_sin_lluvia'])
        & (train_df.loc[mask_neg, 'wind_max']       > CONDICIONES_EXTREMAS['wind_max'])
        & (train_df.loc[mask_neg, 'swvl1_aprox']    < CONDICIONES_EXTREMAS['swvl1_aprox'])
    )
    neg_extremos = train_df[mask_neg][mask_extremo].copy()
    print(f"  Negativos originales en train: {mask_neg.sum()}")
    print(f"  Negativos con condiciones extremas sin fuego: {len(neg_extremos)}")

    if len(neg_extremos) > 0:
        neg_extremos_dup = pd.concat([neg_extremos] * FACTOR_DUPLICACION, ignore_index=True)
        train_df = pd.concat([train_df, neg_extremos_dup], ignore_index=True)
        print(f"  Train después de duplicar extremos: {len(train_df)}")
    else:
        print("  ⚠️ No se encontraron negativos con condiciones extremas.")


# ──────────────────────────────────────────────────────────────────
# 12. Features avanzadas
# ──────────────────────────────────────────────────────────────────
print("Creando features avanzadas...")
for name, dset in [('train', train_df), ('val', val_df), ('test', test_df)]:
    dset = dset.sort_values(['lat_era5', 'lon_era5', 'fecha'])
    dset['t2m_max_3d_avg'] = dset.groupby(['lat_era5', 'lon_era5'])['t2m_max'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    dset['precip_3d_sum'] = dset.groupby(['lat_era5', 'lon_era5'])['precip'].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    dset['rayos_ndvi'] = dset['tasa_rayos'] * dset['ndvi_anomalia_aprox'].clip(lower=0)
    dset['temp_ndvi']  = dset['t2m_anomalia_aprox'] * dset['ndvi_anomalia_aprox'].clip(lower=0)
    dset['fire_danger'] = (
        (dset['t2m_max'] - 20).clip(lower=0)
        * (dset['wind_max'] / 10)
        * (1 - dset['swvl1_aprox'])
    )
    if name == 'train':   train_df = dset
    elif name == 'val':   val_df   = dset
    else:                 test_df  = dset


# ──────────────────────────────────────────────────────────────────
# 13. Fuegos activos lag 1
# ──────────────────────────────────────────────────────────────────
print("Integrando fuegos_activos_lag1...")
fuegos_hist = pd.read_parquet(ERA5_DIR / "fuegos_activos_historicos.parquet")
fuegos_hist['fecha'] = pd.to_datetime(fuegos_hist['fecha'])

celdas_monitoreadas = top_celdas[['lat_era5', 'lon_era5']]

for name, dset in [('train', train_df), ('val', val_df), ('test', test_df)]:
    dset = dset.merge(fuegos_hist, on=['lat_era5', 'lon_era5', 'fecha'], how='left')
    dset['fuegos_activos'] = dset['fuegos_activos'].fillna(0).astype(int)
    dset = dset.sort_values(['lat_era5', 'lon_era5', 'fecha'])
    dset['fuegos_activos_lag1'] = (dset
                                   .groupby(['lat_era5', 'lon_era5'])['fuegos_activos']
                                   .shift(1)
                                   .fillna(0)
                                   .astype(int))
    dset.drop(columns=['fuegos_activos'], inplace=True)
    if name == 'train':   train_df = dset
    elif name == 'val':   val_df   = dset
    else:                 test_df  = dset


# ──────────────────────────────────────────────────────────────────
# 14. Features finales
# ──────────────────────────────────────────────────────────────────
FULL_FEATURES = [
    't2m_max', 't2m_mean', 't2m_anomalia_aprox', 'wind_max', 'wind_dir',
    'precip', 'dias_sin_lluvia', 'sp_min', 'swvl1_aprox',
    'ndvi_lag15', 'ndvi_anomalia_aprox', 'sin_dia', 'cos_dia',
    'densidad_poblacional', 'es_feriado', 'es_finde_semana', 'tasa_rayos',
    't2m_max_3d_avg', 'precip_3d_sum', 'rayos_ndvi', 'temp_ndvi', 'fire_danger',
    'fuegos_activos_lag1', 'hr_mean', 'temp_hr', 'fwi', 'isi', 'bui',
    'pressure_gradient', 'hr_min_chile', 'wind_gust_max_chile',
    'pressure_gradient_zonda', 'altitud'
]
print(f"Features totales: {len(FULL_FEATURES)}")

train_df = train_df.dropna(subset=FULL_FEATURES)
val_df   = val_df.dropna(subset=FULL_FEATURES)
test_df  = test_df.dropna(subset=FULL_FEATURES)

X_train, y_train = train_df[FULL_FEATURES], train_df['fuego']
X_val,   y_val   = val_df[FULL_FEATURES],   val_df['fuego']
X_test,  y_test  = test_df[FULL_FEATURES],  test_df['fuego']

print(f"\nTrain (≤2016): {X_train.shape} | Pos={y_train.sum()} | Neg={len(y_train)-y_train.sum()}")
print(f"Val (2017-18): {X_val.shape}   | Pos={y_val.sum()}   | Neg={len(y_val)-y_val.sum()}")
print(f"Test (≥2019):  {X_test.shape}  | Pos={y_test.sum()}  | Neg={len(y_test)-y_test.sum()}")


# ──────────────────────────────────────────────────────────────────
# 15. Entrenamiento
# ──────────────────────────────────────────────────────────────────
# CAMBIO v2: scale_pos_weight se calcula DESPUÉS del enriquecimiento de negativos.
# El enriquecimiento agrega negativos extremos (FACTOR_DUPLICACION=3), lo que
# cambia la proporción real que XGBoost debe compensar.
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
print(f"\nEntrenando XGBoost con scale_pos_weight={scale_pos_weight:.2f} "
      f"(calculado post-enriquecimiento)...")

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    eval_metric='auc',
    use_label_encoder=False,
    tree_method='hist',
    early_stopping_rounds=30,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)


# ──────────────────────────────────────────────────────────────────
# 16. Evaluación
# ──────────────────────────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]
auc    = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print(f"\n{'='*55}")
print(f"AUC-ROC: {auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")
print(f"\nBarrido de umbrales sobre test:")
print(f"{'Umbral':>8} | {'Precisión':>10} | {'Recall':>8} | {'F1':>6} | {'Alertas':>10}")
for u in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    yp = (y_prob >= u).astype(int)
    p  = precision_score(y_test, yp, zero_division=0)
    r  = recall_score(y_test, yp, zero_division=0)
    f  = f1_score(y_test, yp, zero_division=0)
    n  = int(yp.sum())
    print(f"{u:>8.2f} | {p:>10.3f} | {r:>8.3f} | {f:>6.3f} | {n:>10}")
print(f"{'='*55}")

feat_imp = pd.Series(model.feature_importances_, index=FULL_FEATURES)
print("\nTop 10 features por importancia:")
print(feat_imp.sort_values(ascending=False).head(10))

# Distribución de probabilidades por mes (útil para detectar si el modelo
# está siendo sensato estacionalmente: invierno → probs bajas, verano → altas)
print("\nDistribución de probabilidad media por mes (sobre test):")
test_df_eval = test_df.copy()
test_df_eval['prob'] = y_prob
prob_mes = (test_df_eval
            .groupby(test_df_eval['fecha'].dt.month)['prob']
            .agg(['mean', 'max', 'count'])
            .rename(columns={'mean': 'prob_media', 'max': 'prob_max', 'count': 'n'}))
print(prob_mes.to_string())


# ──────────────────────────────────────────────────────────────────
# 17. Guardar modelo con metadatos
# ──────────────────────────────────────────────────────────────────
# CAMBIO v2: el calibrador NO se guarda.
# Razón: el calibrador Platt se ajusta sobre val (tasa de positivos ~15%)
# pero en producción la tasa real es cercana a cero (se predice una fecha
# futura sobre todas las celdas, no sobre un dataset curado). La calibración
# comprime las probabilidades hacia abajo (aciertos reales ~0.23 en vez de
# ~0.54) sin mejorar AUC/PR-AUC. alerta_incendios.py usa USAR_CALIBRACION=False.
# Si en el futuro se quiere calibrar, recalcular el calibrador con una muestra
# que refleje la tasa base real de producción antes de activarlo.

model_data = {
    'modelo':             model,
    'features':           FULL_FEATURES,
    'auc_test':           float(auc),
    'pr_auc_test':        float(pr_auc),
    'version':            VERSION_MODELO,
    'fecha_entrenamiento': pd.Timestamp.now().isoformat(),
    'train_hasta_anio':   2016,
    'val_anios':          '2017-2018',
    'test_desde_anio':    2019,
    'n_celdas_entrenamiento': int(len(top_celdas)),
    'scale_pos_weight_usado': float(scale_pos_weight),
    'umbral_provisional': UMBRAL_UNICO_PROVISIONAL,
    # El calibrador se omite deliberadamente (ver comentario arriba).
    # Para regenerarlo: sklearn LogisticRegression sobre y_val_prob → y_val.
}

output_path = ERA5_DIR / "modelo_xgb_mejorado.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n✅ Artefactos guardados:")
print(f"  - {output_path}  (versión {VERSION_MODELO}, sin calibrador)")
print(f"  - top_celdas.parquet          ({len(top_celdas)} celdas)")
print(f"  - umbrales_celda.parquet      (umbral único {UMBRAL_UNICO_PROVISIONAL})")
print(f"  - altitud_celdas.parquet")
print(f"  - climatologia_t2m.parquet")
print(f"  - climatologia_ndvi.parquet")
print(f"\nMetadatos del modelo:")
for k, v in model_data.items():
    if k not in ('modelo', 'features'):
        print(f"  {k}: {v}")

