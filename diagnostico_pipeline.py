#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
diagnostico_pipeline.py
=======================
Experimento diagnóstico para verificar si el pipeline de inferencia
produce probabilidades esperadas con features sintéticas de temporada alta.

Pregunta central:
  ¿El modelo es capaz de producir probabilidades altas (>0.35) cuando
  las condiciones meteorológicas son las de un día típico de riesgo?

Si la respuesta es SÍ → el pipeline está bien y las probs bajas en
producción se explican por las condiciones reales actuales.

Si la respuesta es NO → hay algo roto en la cadena de features
(unidades incorrectas, columna desplazada, escala distinta, etc.)

Estructura del experimento:
  1. Cargar el modelo y las features esperadas.
  2. Construir 4 vectores sintéticos que cubren el espectro de riesgo:
       - Perfil A: condiciones de invierno típico (debería dar prob baja)
       - Perfil B: condiciones de zonda moderado (debería dar prob media)
       - Perfil C: condiciones de zonda intenso (debería dar prob alta)
       - Perfil D: condiciones extremas + foco activo (debería dar prob muy alta)
  3. Aplicar cada vector a las 10 celdas con mayor historial de incendios
     (las más conocidas por el modelo).
  4. Comparar las probabilidades obtenidas con las esperadas según el
     backtesting histórico para esos mismos meses.
  5. Verificar feature por feature que los valores estén en el rango
     visto durante el entrenamiento.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

ERA5_DIR = Path("era5_salta")

# ──────────────────────────────────────────────────────────────────
# 1. Cargar modelo
# ──────────────────────────────────────────────────────────────────
print("Cargando modelo...")
with open(ERA5_DIR / "modelo_xgb_mejorado.pkl", "rb") as f:
    model_data = pickle.load(f)

model    = model_data['modelo']
features = model_data['features']

print(f"  Versión:             {model_data.get('version', '?')}")
print(f"  Fecha entrenamiento: {model_data.get('fecha_entrenamiento', '?')}")
print(f"  AUC test:            {model_data.get('auc_test', '?'):.4f}")
print(f"  Features esperadas:  {len(features)}")


# ──────────────────────────────────────────────────────────────────
# 2. Cargar tablas auxiliares
# ──────────────────────────────────────────────────────────────────
top_celdas  = pd.read_parquet(ERA5_DIR / "top_celdas.parquet")
alt_celdas  = pd.read_parquet(ERA5_DIR / "altitud_celdas.parquet")
clim_t2m    = pd.read_parquet(ERA5_DIR / "climatologia_t2m.parquet")
clim_ndvi   = pd.read_parquet(ERA5_DIR / "climatologia_ndvi.parquet")
rayos_df    = pd.read_parquet(ERA5_DIR / "rayos_celdas.parquet")

# Las 10 celdas más incendiadas históricamente
top10 = top_celdas.nlargest(10, 'frecuencia')[['lat_era5', 'lon_era5', 'frecuencia', 'departamento']].copy()
top10 = top10.merge(alt_celdas, on=['lat_era5', 'lon_era5'], how='left')
top10['altitud'] = top10['altitud'].fillna(1000)

print(f"\nTop 10 celdas (mayor historial de incendios):")
print(top10[['lat_era5', 'lon_era5', 'departamento', 'frecuencia', 'altitud']].to_string(index=False))


# ──────────────────────────────────────────────────────────────────
# 3. Definir perfiles de riesgo
# ──────────────────────────────────────────────────────────────────
# Julio en Salta: temporada de zonda.
# Valores de referencia del backtesting para celdas históricas en julio:
#   prob_media ≈ 0.49,  prob_max ≈ 0.91
#
# Los perfiles usan julio (mes=7, dia_año=196) como contexto temporal.
# Cada perfil representa un escenario meteorológico distinto.

MES        = 7       # julio
DIA_AÑO    = 196     # ~15 de julio
SIN_DIA    = np.sin(2 * np.pi * DIA_AÑO / 365.25)
COS_DIA    = np.cos(2 * np.pi * DIA_AÑO / 365.25)

# Notas sobre unidades (deben coincidir exactamente con entrenamiento):
#   wind_max: m/s (no km/h — se divide por 3.6 en alerta_incendios.py)
#   sp_min:   Pa  (se multiplica por 100 en alerta_incendios.py)
#   pressure_gradient: Pa (presion_chile_hpa * 100 - sp_min)

PERFILES = {
    'A_invierno_tipico': {
        # Condiciones frescas, algo de lluvia reciente, sin viento zonda
        # Expectativa: prob BAJA (< 0.20)
        't2m_max':              18.0,
        't2m_mean':             10.0,
        't2m_anomalia_aprox':   -2.0,   # más frío de lo normal
        'wind_max':              2.0,   # m/s — viento suave
        'wind_dir':            180.0,
        'precip':                2.0,   # lluvia hoy
        'dias_sin_lluvia':       2.0,
        'sp_min':           85000.0,    # Pa — presión normal de altura
        'swvl1_aprox':           0.20,  # suelo húmedo
        'ndvi_lag15':            0.40,
        'ndvi_anomalia_aprox':   0.05,  # vegetación normal
        'fuegos_activos_lag1':   0,
        'hr_mean':              60.0,
        'temp_hr':               7.2,   # t2m_max * (100-hr)/100
        'fwi':                   2.0,
        'isi':                   0.5,
        'bui':                  15.0,
        'pressure_gradient':  5000.0,   # Pa — gradiente bajo
        'hr_min_chile':         50.0,
        'wind_gust_max_chile':   3.0,   # m/s
        'pressure_gradient_zonda': 5000.0,
        'densidad_poblacional':  5.0,
        'es_feriado':            0,
        'es_finde_semana':       0,
        'tasa_rayos':            0.05,
        't2m_max_3d_avg':       17.0,
        'precip_3d_sum':         4.0,
        'rayos_ndvi':            0.002,
        'temp_ndvi':            -0.10,
        'fire_danger':           0.5,   # (18-20).clip(0) * (2/10) * (1-0.20) → 0
    },

    'B_zonda_moderado': {
        # Zonda moderado: t alta, sin lluvia, viento del oeste, baja humedad
        # Expectativa: prob MEDIA (0.25 – 0.45)
        't2m_max':              32.0,
        't2m_mean':             22.0,
        't2m_anomalia_aprox':   8.0,    # muy por encima de la climatología
        'wind_max':              7.0,   # m/s — viento fuerte
        'wind_dir':            270.0,   # del oeste (zonda)
        'precip':                0.0,
        'dias_sin_lluvia':      10.0,
        'sp_min':           82000.0,    # Pa — presión baja (foehn)
        'swvl1_aprox':           0.09,  # suelo seco
        'ndvi_lag15':            0.30,
        'ndvi_anomalia_aprox':  -0.10,  # vegetación más seca de lo normal
        'fuegos_activos_lag1':   0,
        'hr_mean':              25.0,
        'temp_hr':              24.0,   # 32 * (100-25)/100
        'fwi':                  25.0,
        'isi':                   4.0,
        'bui':                  80.0,
        'pressure_gradient':  18000.0,  # Pa — gradiente Chile-Salta
        'hr_min_chile':         15.0,
        'wind_gust_max_chile':  12.0,   # m/s
        'pressure_gradient_zonda': 15000.0,
        'densidad_poblacional':  5.0,
        'es_feriado':            0,
        'es_finde_semana':       0,
        'tasa_rayos':            0.10,
        't2m_max_3d_avg':       30.0,
        'precip_3d_sum':         0.0,
        'rayos_ndvi':            0.010,
        'temp_ndvi':             0.80,
        'fire_danger':           8.4,   # (32-20) * (7/10) * (1-0.09)
    },

    'C_zonda_intenso': {
        # Zonda intenso: condiciones extremas, 15+ días sin lluvia
        # Expectativa: prob ALTA (0.45 – 0.70)
        't2m_max':              38.0,
        't2m_mean':             27.0,
        't2m_anomalia_aprox':  13.0,
        'wind_max':             10.0,   # m/s — viento muy fuerte
        'wind_dir':            265.0,
        'precip':                0.0,
        'dias_sin_lluvia':      18.0,
        'sp_min':           80000.0,    # Pa — presión muy baja
        'swvl1_aprox':           0.07,  # suelo extremadamente seco
        'ndvi_lag15':            0.20,
        'ndvi_anomalia_aprox':  -0.20,  # vegetación muy seca
        'fuegos_activos_lag1':   0,
        'hr_mean':              15.0,
        'temp_hr':              32.3,   # 38 * (100-15)/100
        'fwi':                  50.0,
        'isi':                   8.0,
        'bui':                 150.0,
        'pressure_gradient':  30000.0,
        'hr_min_chile':          8.0,
        'wind_gust_max_chile':  18.0,   # m/s
        'pressure_gradient_zonda': 28000.0,
        'densidad_poblacional':  5.0,
        'es_feriado':            0,
        'es_finde_semana':       1,     # fin de semana: más actividad humana
        'tasa_rayos':            0.20,
        't2m_max_3d_avg':       36.0,
        'precip_3d_sum':         0.0,
        'rayos_ndvi':            0.040,
        'temp_ndvi':             2.60,
        'fire_danger':          27.9,   # (38-20) * (10/10) * (1-0.07)
    },

    'D_extremo_con_foco': {
        # Condiciones extremas + foco activo ayer (fuegos_activos_lag1=1)
        # Expectativa: prob MUY ALTA (> 0.70)
        # Este es el escenario que el modelo debería reconocer con certeza.
        't2m_max':              40.0,
        't2m_mean':             29.0,
        't2m_anomalia_aprox':  15.0,
        'wind_max':             12.0,
        'wind_dir':            260.0,
        'precip':                0.0,
        'dias_sin_lluvia':      21.0,
        'sp_min':           79000.0,
        'swvl1_aprox':           0.06,
        'ndvi_lag15':            0.15,
        'ndvi_anomalia_aprox':  -0.25,
        'fuegos_activos_lag1':   1,     # ← la feature más importante: foco activo ayer
        'hr_mean':              12.0,
        'temp_hr':              35.2,
        'fwi':                  70.0,
        'isi':                  12.0,
        'bui':                 200.0,
        'pressure_gradient':  40000.0,
        'hr_min_chile':          5.0,
        'wind_gust_max_chile':  22.0,
        'pressure_gradient_zonda': 38000.0,
        'densidad_poblacional':  5.0,
        'es_feriado':            0,
        'es_finde_semana':       0,
        'tasa_rayos':            0.30,
        't2m_max_3d_avg':       39.0,
        'precip_3d_sum':         0.0,
        'rayos_ndvi':            0.075,
        'temp_ndvi':             3.75,
        'fire_danger':          35.0,   # (40-20) * (12/10) * (1-0.06)
    },
}

EXPECTATIVAS = {
    'A_invierno_tipico':   '< 0.20  (prob baja)',
    'B_zonda_moderado':    '0.25 – 0.45  (prob media)',
    'C_zonda_intenso':     '0.45 – 0.70  (prob alta)',
    'D_extremo_con_foco':  '> 0.70  (prob muy alta)',
}


# ──────────────────────────────────────────────────────────────────
# 4. Construir vectores por celda y predecir
# ──────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("EXPERIMENTO: probabilidades del modelo con features sintéticas")
print(f"Contexto temporal: julio (mes={MES}, dia_año={DIA_AÑO})")
print(f"{'='*65}")

resultados = []

for nombre_perfil, perfil_base in PERFILES.items():
    filas = []
    for _, celda in top10.iterrows():
        lat = celda['lat_era5']
        lon = celda['lon_era5']

        # Climatología t2m para esta celda en julio
        clim_t = clim_t2m[
            (clim_t2m['lat_era5'] == lat) &
            (clim_t2m['lon_era5'] == lon) &
            (clim_t2m['mes'] == MES)
        ]
        t2m_clim_val = clim_t['t2m_clim'].values[0] if len(clim_t) > 0 else 20.0

        # Climatología NDVI para esta celda en julio
        clim_n = clim_ndvi[
            (clim_ndvi['lat_era5'] == lat) &
            (clim_ndvi['lon_era5'] == lon) &
            (clim_ndvi['mes'] == MES)
        ]
        ndvi_clim_val = clim_n['ndvi_clim'].values[0] if len(clim_n) > 0 else 0.30

        # Tasa de rayos para esta celda en julio
        rayos_row = rayos_df[
            (rayos_df['lat_era5'] == lat) &
            (rayos_df['lon_era5'] == lon) &
            (rayos_df['dia_año'] == DIA_AÑO)
        ]
        tasa_rayos_val = rayos_row['tasa_rayos'].values[0] if len(rayos_row) > 0 else perfil_base['tasa_rayos']

        fila = dict(perfil_base)  # copiar el perfil base

        # Sobreescribir con valores específicos de la celda
        fila['altitud']     = celda['altitud']
        fila['sin_dia']     = SIN_DIA
        fila['cos_dia']     = COS_DIA
        fila['tasa_rayos']  = tasa_rayos_val

        # Recalcular anomalía t2m con la climatología real de la celda
        fila['t2m_anomalia_aprox'] = fila['t2m_mean'] - t2m_clim_val

        # Recalcular anomalía NDVI con la climatología real de la celda
        fila['ndvi_anomalia_aprox'] = fila['ndvi_lag15'] - ndvi_clim_val

        # Recalcular features derivadas que dependen de anomalías
        fila['rayos_ndvi'] = tasa_rayos_val * max(fila['ndvi_anomalia_aprox'], 0)
        fila['temp_ndvi']  = fila['t2m_anomalia_aprox'] * max(fila['ndvi_anomalia_aprox'], 0)

        fila['lat']  = lat
        fila['lon']  = lon
        fila['depto'] = celda['departamento']
        filas.append(fila)

    df_perfil = pd.DataFrame(filas)

    # Verificar que todas las features del modelo estén presentes
    missing = set(features) - set(df_perfil.columns)
    if missing:
        print(f"  ⚠️ Features faltantes en perfil {nombre_perfil}: {missing}")
        for col in missing:
            df_perfil[col] = 0

    X = df_perfil[features]
    assert list(X.columns) == features, "Orden de features no coincide con el modelo"
    probas = model.predict_proba(X)[:, 1]

    df_perfil['probabilidad'] = probas

    prob_min  = probas.min()
    prob_mean = probas.mean()
    prob_max  = probas.max()

    print(f"\n{'─'*65}")
    print(f"Perfil: {nombre_perfil}")
    print(f"  Expectativa:   {EXPECTATIVAS[nombre_perfil]}")
    print(f"  Prob mínima:   {prob_min:.4f}")
    print(f"  Prob media:    {prob_mean:.4f}")
    print(f"  Prob máxima:   {prob_max:.4f}")

    # Diagnóstico por celda
    print(f"  Por celda:")
    for _, row in df_perfil.iterrows():
        marca = " ◀ máx" if row['probabilidad'] == prob_max else ""
        print(f"    {row['depto']:25s}  prob={row['probabilidad']:.4f}{marca}")

    resultados.append({
        'perfil':    nombre_perfil,
        'expectativa': EXPECTATIVAS[nombre_perfil],
        'prob_min':  round(prob_min, 4),
        'prob_mean': round(prob_mean, 4),
        'prob_max':  round(prob_max, 4),
    })


# ──────────────────────────────────────────────────────────────────
# 5. Resumen y diagnóstico
# ──────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("RESUMEN")
print(f"{'='*65}")
df_res = pd.DataFrame(resultados)
print(df_res.to_string(index=False))

print(f"\n{'='*65}")
print("DIAGNÓSTICO DEL PIPELINE")
print(f"{'='*65}")

prob_A = df_res.loc[df_res['perfil'] == 'A_invierno_tipico',  'prob_mean'].values[0]
prob_B = df_res.loc[df_res['perfil'] == 'B_zonda_moderado',   'prob_mean'].values[0]
prob_C = df_res.loc[df_res['perfil'] == 'C_zonda_intenso',    'prob_mean'].values[0]
prob_D = df_res.loc[df_res['perfil'] == 'D_extremo_con_foco', 'prob_mean'].values[0]

ok_A = prob_A < 0.20
ok_B = 0.15 < prob_B < 0.60
ok_C = prob_C > 0.30
ok_D = prob_D > 0.50

print(f"  A (invierno típico)   → {prob_A:.4f}  {'✅ OK' if ok_A else '❌ PROBLEMA: debería ser < 0.20'}")
print(f"  B (zonda moderado)    → {prob_B:.4f}  {'✅ OK' if ok_B else '❌ PROBLEMA: debería estar entre 0.15 y 0.60'}")
print(f"  C (zonda intenso)     → {prob_C:.4f}  {'✅ OK' if ok_C else '❌ PROBLEMA: debería ser > 0.30'}")
print(f"  D (extremo+foco)      → {prob_D:.4f}  {'✅ OK' if ok_D else '❌ PROBLEMA: debería ser > 0.50'}")

todos_ok = ok_A and ok_B and ok_C and ok_D

print()
if todos_ok:
    print("✅ PIPELINE OK: el modelo responde correctamente a los perfiles de riesgo.")
    print("   Las probabilidades bajas en producción se explican por las condiciones")
    print("   reales actuales, no por un problema en el pipeline de inferencia.")
    print()
    print("   Siguiente paso: verificar si fuegos_activos_lag1 está siendo")
    print("   correctamente alimentado desde FIRMS en las corridas recientes.")
    print("   Comparar prob_C vs prod del 26/06 (prob_max=0.44 con fuegos_lag1=0).")
else:
    print("❌ PIPELINE CON PROBLEMAS: algún perfil no produce la probabilidad esperada.")
    print("   Verificar:")
    print("   1. Unidades de wind_max (debe ser m/s, no km/h).")
    print("   2. Unidades de sp_min (debe ser Pa, no hPa).")
    print("   3. Orden de features (assert debería haber fallado si hay discrepancia).")
    print("   4. Versión del modelo cargado (ver metadatos arriba).")
    print("   5. Si prob_D es baja con fuegos_activos_lag1=1, el modelo puede haber")
    print("      aprendido una distribución muy distinta de esa feature.")

# ──────────────────────────────────────────────────────────────────
# 6. Test adicional: sensibilidad a fuegos_activos_lag1
# ──────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("TEST DE SENSIBILIDAD: impacto de fuegos_activos_lag1")
print("(usando perfil C como base, variando solo esta feature)")
print(f"{'='*65}")

# Tomar la primera celda del top10 como representativa
celda_ref = top10.iloc[0]
lat_ref   = celda_ref['lat_era5']
lon_ref   = celda_ref['lon_era5']

clim_t_ref = clim_t2m[
    (clim_t2m['lat_era5'] == lat_ref) &
    (clim_t2m['lon_era5'] == lon_ref) &
    (clim_t2m['mes'] == MES)
]
t2m_clim_ref = clim_t_ref['t2m_clim'].values[0] if len(clim_t_ref) > 0 else 20.0

clim_n_ref = clim_ndvi[
    (clim_ndvi['lat_era5'] == lat_ref) &
    (clim_ndvi['lon_era5'] == lon_ref) &
    (clim_ndvi['mes'] == MES)
]
ndvi_clim_ref = clim_n_ref['ndvi_clim'].values[0] if len(clim_n_ref) > 0 else 0.30

base_C = dict(PERFILES['C_zonda_intenso'])
base_C['altitud']              = celda_ref['altitud']
base_C['sin_dia']              = SIN_DIA
base_C['cos_dia']              = COS_DIA
base_C['t2m_anomalia_aprox']   = base_C['t2m_mean'] - t2m_clim_ref
base_C['ndvi_anomalia_aprox']  = base_C['ndvi_lag15'] - ndvi_clim_ref
base_C['rayos_ndvi']           = base_C['tasa_rayos'] * max(base_C['ndvi_anomalia_aprox'], 0)
base_C['temp_ndvi']            = base_C['t2m_anomalia_aprox'] * max(base_C['ndvi_anomalia_aprox'], 0)

print(f"  Celda: {celda_ref['departamento']} ({lat_ref}, {lon_ref})")
print(f"  Perfil base: C (zonda intenso, 18 días sin lluvia, FWI=50)")
print()
print(f"  {'fuegos_activos_lag1':25s}  probabilidad")

for n_focos in [0, 1, 2, 3, 5]:
    fila = dict(base_C)
    fila['fuegos_activos_lag1'] = n_focos
    df_single = pd.DataFrame([fila])
    missing = set(features) - set(df_single.columns)
    for col in missing:
        df_single[col] = 0
    prob = model.predict_proba(df_single[features])[:, 1][0]
    delta = ""
    if n_focos > 0:
        delta = f"  (+{prob - proba_0:.4f} vs. 0 focos)" if n_focos > 0 and 'proba_0' in dir() else ""
    if n_focos == 0:
        proba_0 = prob
    else:
        delta = f"  (+{prob - proba_0:.4f} vs. 0 focos)"
    print(f"  {n_focos:<25}  {prob:.4f}{delta}")

print()
print("  Si el salto entre 0 y 1 foco es grande (>0.15), confirma que")
print("  fuegos_activos_lag1 es el driver principal y que las corridas")
print("  de producción recientes con lag1=0 producen probs artificialmente bajas.")

