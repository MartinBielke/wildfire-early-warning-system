#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
diagnostico_vectores_reales.py
===============================
Segundo diagnóstico, más confiable que diagnostico_pipeline.py.

El primer diagnóstico (vectores sintéticos) dio resultados extraños: el
perfil "zonda moderado" produjo probabilidad MENOR que "invierno típico",
y ni siquiera el perfil extremo con foco activo superó 0.25. Esto podía
deberse a dos causas distintas:

  (a) El pipeline de inferencia tiene un bug real.
  (b) Los vectores sintéticos tenían combinaciones de features fuera de
      la distribución que el modelo vio en entrenamiento (p. ej. sp_min
      de 80000 Pa para celdas que en realidad rondan 95000-97000 Pa a
      esa altitud), y el árbol simplemente no extrapola bien ahí.

Este script evita ese problema: en lugar de inventar valores, toma
VECTORES REALES del período de test (2019-2024) que el propio modelo
ya calificó con alta probabilidad en julio, y verifica dos cosas:

  1. Que el modelo reproduce esas probabilidades altas de forma
     consistente (confirma que predict_proba() y el pickle están sanos).
  2. Compara la distribución de esas features reales de "julio de alto
     riesgo" contra las features que alerta_incendios_v2.py generó en
     su corrida más reciente (30/06/2026), para ver en qué variable(s)
     está la diferencia real.

Requiere los mismos artefactos que backtesting_v2.py.
"""

import pandas as pd
import numpy as np
import pickle
import requests
from pathlib import Path

ERA5_DIR = Path("era5_salta")

# ──────────────────────────────────────────────────────────────────
# Valores observados en la corrida de producción del 30/06/2026
# (copiados del diagnóstico impreso por alerta_incendios_v2.py).
# Sirven de referencia para comparar contra los vectores reales de julio.
# ──────────────────────────────────────────────────────────────────
PRODUCCION_30_06 = {
    't2m_max':                 {'min': -4.90,   'mean': 15.01,    'max': 28.80},
    'wind_max':                {'min': 0.94,    'mean': 3.82,     'max': 10.97},
    'dias_sin_lluvia':         {'min': 0.00,    'mean': 5.05,     'max': 7.00},
    'swvl1_aprox':             {'min': 0.08,    'mean': 0.09,     'max': 0.20},
    'fwi':                     {'min': 0.00,    'mean': 3.81,     'max': 23.92},
    'isi':                     {'min': 0.00,    'mean': 0.76,     'max': 5.52},
    'bui':                     {'min': 0.00,    'mean': 41.19,    'max': 235.48},
    'pressure_gradient':       {'min': 2270.0,  'mean': 15716.77, 'max': 46390.0},
    'pressure_gradient_zonda': {'min': 12130.0, 'mean': 12130.0,  'max': 12130.0},
    'hr_min_chile':            {'min': 5.00,    'mean': 44.44,    'max': 85.00},
    'wind_gust_max_chile':     {'min': 3.00,    'mean': 9.61,     'max': 31.61},
}
PROB_MAX_PRODUCCION_30_06 = 0.4380


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
# 1. Cargar modelo
# ──────────────────────────────────────────────────────────────────
print("Cargando modelo...")
with open(ERA5_DIR / "modelo_xgb_mejorado.pkl", "rb") as f:
    model_data = pickle.load(f)
model    = model_data['modelo']
features = model_data['features']
print(f"  Versión: {model_data.get('version', '?')}  |  AUC test: {model_data.get('auc_test', 0):.4f}")


# ──────────────────────────────────────────────────────────────────
# 2. Reconstruir el dataset de test (2019-2024) con el mismo pipeline
#    de features que entrenamiento_xgb_v2.py / backtesting_v2.py
# ──────────────────────────────────────────────────────────────────
print("\nCargando datos base...")
top_celdas = pd.read_parquet(ERA5_DIR / "top_celdas.parquet")
celdas_monitoreadas = top_celdas[['lat_era5', 'lon_era5']]

alt_celdas = pd.read_parquet(ERA5_DIR / "altitud_celdas.parquet")
alt_dict = alt_celdas.set_index(['lat_era5', 'lon_era5'])['altitud'].to_dict()

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

# Contexto desde 2018-12-01 para que los lags tengan historia real,
# igual que hace backtesting_v2.py, pero cubriendo TODO el período de
# test (2019-2024) en una sola pasada.
fecha_inicio_contexto = pd.Timestamp('2018-12-01')
fecha_inicio_eval     = pd.Timestamp('2019-01-01')
df = df[df['fecha'] >= fecha_inicio_contexto].copy()
print(f"Período cargado: {df['fecha'].min().date()} a {df['fecha'].max().date()}")

print("Cargando datos de Chile y cordillera...")
chile_data = pd.read_parquet(ERA5_DIR / "datos_chile.parquet")
df = df.merge(chile_data, on='fecha', how='left')
df['pressure_gradient'] = (df['presion_chile_hpa'] * 100) - df['sp_min']
df['pressure_gradient'] = df['pressure_gradient'].fillna(0)
df['hr_min_chile'] = df['hr_min_chile'].fillna(df['hr_min_chile'].median())
df['wind_gust_max_chile'] = df['wind_gust_max_chile'].fillna(0)

cord_data = pd.read_parquet(ERA5_DIR / "presion_cordillera.parquet")
df = df.merge(cord_data, on='fecha', how='left')
df['presion_cordillera_hpa'] = df['presion_cordillera_hpa'].fillna(0)
df['pressure_gradient_zonda'] = (df['presion_chile_hpa'] - df['presion_cordillera_hpa']) * 100
df['pressure_gradient_zonda'] = df['pressure_gradient_zonda'].fillna(0)

df['altitud'] = df.apply(lambda row: alt_dict.get((row['lat_era5'], row['lon_era5']), 1000), axis=1)

print("Calculando features...")
df['dia_año'] = df['fecha'].dt.dayofyear
df['sin_dia'] = np.sin(2 * np.pi * df['dia_año'] / 365.25)
df['cos_dia'] = np.cos(2 * np.pi * df['dia_año'] / 365.25)
df['mes'] = df['fecha'].dt.month

clim_t2m = pd.read_parquet(ERA5_DIR / "climatologia_t2m.parquet")
df = df.merge(clim_t2m, on=['lat_era5', 'lon_era5', 'mes'], how='left')
df['t2m_anomalia_aprox'] = df['t2m_mean'] - df['t2m_clim']

df = df.sort_values(['lat_era5', 'lon_era5', 'fecha'])
df['ndvi_lag15'] = df.groupby(['lat_era5', 'lon_era5'])['ndvi'].shift(15)
df['ndvi_lag15'] = df.groupby(['lat_era5', 'lon_era5'])['ndvi_lag15'].ffill(limit=30)
df['ndvi_lag15'] = df['ndvi_lag15'].fillna(df['ndvi'].mean())

clim_ndvi = pd.read_parquet(ERA5_DIR / "climatologia_ndvi.parquet")
df = df.merge(clim_ndvi, on=['lat_era5', 'lon_era5', 'mes'], how='left')
df['ndvi_anomalia_aprox'] = df['ndvi_lag15'] - df['ndvi_clim']

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

dens_pob = pd.read_parquet(ERA5_DIR / "densidad_poblacional_celdas.parquet")
df = df.merge(dens_pob, on=['lat_era5', 'lon_era5'], how='left')
df['densidad_poblacional'] = df['densidad_poblacional'].fillna(0)

rayos_df = pd.read_parquet(ERA5_DIR / "rayos_celdas.parquet")
df = df.merge(rayos_df, on=['lat_era5', 'lon_era5', 'dia_año'], how='left')
df['tasa_rayos'] = df['tasa_rayos'].fillna(0)

years = df['fecha'].dt.year.unique()
all_holidays = []
for year in years:
    all_holidays.extend(obtener_feriados_argentina(year))
holidays_set = set(all_holidays)
df['es_feriado'] = df['fecha'].dt.strftime('%Y-%m-%d').isin(holidays_set).astype(int)
df['es_finde_semana'] = (df['fecha'].dt.dayofweek >= 5).astype(int)

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

fuegos_hist = pd.read_parquet(ERA5_DIR / "fuegos_activos_historicos.parquet")
fuegos_hist = fuegos_hist.merge(celdas_monitoreadas, on=['lat_era5', 'lon_era5'], how='inner')
df = df.merge(fuegos_hist, on=['lat_era5', 'lon_era5', 'fecha'], how='left')
df['fuegos_activos'] = df['fuegos_activos'].fillna(0).astype(int)
df = df.sort_values(['lat_era5', 'lon_era5', 'fecha'])
df['fuegos_activos_lag1'] = df.groupby(['lat_era5', 'lon_era5'])['fuegos_activos'].shift(1).fillna(0).astype(int)

hr_hist = pd.read_parquet(ERA5_DIR / "hr_historico.parquet")
hr_hist['fecha'] = pd.to_datetime(hr_hist['fecha'])
df = df.merge(hr_hist, on=['lat_era5', 'lon_era5', 'fecha'], how='left')
df['hr_mean'] = df['hr_mean'].fillna(df['hr_mean'].median())
df['temp_hr'] = df['t2m_max'] * (100 - df['hr_mean']) / 100

fwi_hist = pd.read_parquet(ERA5_DIR / "fwi_historico.parquet")
fwi_hist['fecha'] = pd.to_datetime(fwi_hist['fecha'])
df = df.merge(fwi_hist[['lat_era5', 'lon_era5', 'fecha', 'fwi', 'isi', 'bui']],
              on=['lat_era5', 'lon_era5', 'fecha'], how='left')
df['fwi'] = df['fwi'].fillna(0)
df['isi'] = df['isi'].fillna(0)
df['bui'] = df['bui'].fillna(0)

# Recortar al período de evaluación real (2019 en adelante)
df_eval = df[df['fecha'] >= fecha_inicio_eval].copy()

missing = set(features) - set(df_eval.columns)
for col in missing:
    df_eval[col] = 0

X_all = df_eval[features]
assert list(X_all.columns) == features, "Orden de features no coincide con el modelo"
df_eval['y_prob'] = model.predict_proba(X_all)[:, 1]

print(f"Registros de test (2019-2024): {len(df_eval)}")


# ──────────────────────────────────────────────────────────────────
# 3. Filtrar julio de alta probabilidad
# ──────────────────────────────────────────────────────────────────
julio = df_eval[df_eval['mes'] == 7].copy()
print(f"\nRegistros de julio en test: {len(julio)}")
print(f"Probabilidad media en julio: {julio['y_prob'].mean():.4f}")
print(f"Probabilidad máxima en julio: {julio['y_prob'].max():.4f}")

julio_alto = julio[julio['y_prob'] > 0.6].copy()
print(f"\nRegistros de julio con probabilidad > 0.6: {len(julio_alto)}")

if len(julio_alto) == 0:
    print("⚠️ No hay registros con prob > 0.6 en julio. Bajando el umbral a 0.4...")
    julio_alto = julio[julio['y_prob'] > 0.4].copy()
    print(f"Registros de julio con probabilidad > 0.4: {len(julio_alto)}")

print(f"\n{'='*70}")
print("CARACTERÍSTICAS DE LOS DÍAS DE JULIO CON ALTA PROBABILIDAD (REALES)")
print(f"{'='*70}")

cols_comparar = ['t2m_max', 'wind_max', 'dias_sin_lluvia', 'swvl1_aprox',
                  'fwi', 'isi', 'bui', 'pressure_gradient',
                  'pressure_gradient_zonda', 'hr_min_chile',
                  'wind_gust_max_chile', 'fuegos_activos_lag1']

stats_julio_alto = {}
for col in cols_comparar:
    if col in julio_alto.columns:
        stats_julio_alto[col] = {
            'min':  julio_alto[col].min(),
            'mean': julio_alto[col].mean(),
            'max':  julio_alto[col].max(),
        }
        print(f"  {col:26s}  min={stats_julio_alto[col]['min']:>10.2f}  "
              f"mean={stats_julio_alto[col]['mean']:>10.2f}  "
              f"max={stats_julio_alto[col]['max']:>10.2f}")


# ──────────────────────────────────────────────────────────────────
# 4. Sanity check: el modelo reproduce las mismas probabilidades
# ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SANITY CHECK: ¿el modelo reproduce las probabilidades sobre estos")
print("mismos vectores reales al volver a predecir?")
print(f"{'='*70}")

X_verif = julio_alto[features]
probas_verif = model.predict_proba(X_verif)[:, 1]
diferencia_max = np.abs(probas_verif - julio_alto['y_prob'].values).max()

print(f"  Diferencia máxima entre predicción original y re-predicción: {diferencia_max:.6f}")
if diferencia_max < 1e-6:
    print("  ✅ El modelo es determinístico y reproduce exactamente las mismas")
    print("     probabilidades. predict_proba() y el pickle funcionan correctamente.")
else:
    print("  ⚠️ Hay diferencia entre corridas — investigar semillas aleatorias o")
    print("     estado del modelo.")

print(f"\n  Ejemplos de vectores reales con alta probabilidad (top 5):")
top5_ejemplos = julio_alto.nlargest(5, 'y_prob')[
    ['lat_era5', 'lon_era5', 'fecha', 'y_prob'] + cols_comparar
]
print(top5_ejemplos.to_string(index=False))


# ──────────────────────────────────────────────────────────────────
# 5. Comparación directa: julio real de alto riesgo vs. producción 30/06
# ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("COMPARACIÓN: JULIO REAL DE ALTO RIESGO  vs.  PRODUCCIÓN 30/06/2026")
print(f"{'='*70}")
print(f"{'Feature':26s}  {'Julio alto (mean)':>18s}  {'Producción (mean)':>18s}  {'Ratio':>8s}")

for col in cols_comparar:
    if col in stats_julio_alto and col in PRODUCCION_30_06:
        v_julio = stats_julio_alto[col]['mean']
        v_prod  = PRODUCCION_30_06[col]['mean']
        ratio = (v_prod / v_julio) if v_julio != 0 else float('nan')
        marca = ""
        if not np.isnan(ratio):
            if ratio < 0.3 or ratio > 3:
                marca = "  ⚠️ gran diferencia"
        print(f"{col:26s}  {v_julio:>18.2f}  {v_prod:>18.2f}  {ratio:>8.2f}{marca}")

print(f"\nProbabilidad máxima en julio real de alto riesgo (test): {julio_alto['y_prob'].max():.4f}")
print(f"Probabilidad máxima observada en producción (30/06/2026): {PROB_MAX_PRODUCCION_30_06:.4f}")

print(f"\n{'='*70}")
print("INTERPRETACIÓN")
print(f"{'='*70}")
print("""
Si las medias de 'Julio alto' y 'Producción' son parecidas (ratio cercano
a 1) mientras que la probabilidad de producción sigue siendo mucho más baja
que la de julio histórico, el problema NO está en los valores de las
features individuales, sino en cómo se combinan o en alguna feature que
no está en esta tabla (revisar ndvi_anomalia_aprox, temp_ndvi, rayos_ndvi,
densidad_poblacional, es_feriado, es_finde_semana, altitud, sin_dia/cos_dia).

Si en cambio hay una o más features con ratio muy distinto de 1 (marcadas
con ⚠️ arriba), esa es la pista concreta: las condiciones reales del 30/06
simplemente no son comparables a un día de julio de alto riesgo histórico.
En ese caso el sistema está funcionando correctamente — el 30 de junio de
2026 fue, meteorológicamente, un día más tranquilo que los julios de alto
riesgo del pasado, y hay que esperar a que las condiciones reales cambien
(o revisar si julio 2026 en particular está siendo un mes atípicamente
húmedo/fresco, lo cual se puede chequear con datos de precipitación
acumulada de la temporada).
""")

