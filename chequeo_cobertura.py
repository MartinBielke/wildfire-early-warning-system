#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
chequeo_cobertura.py
=======================

Chequeo de cobertura territorial del modelo de incendios.

Compara el grid ERA5 completo dentro del límite de Salta contra las celdas
que tienen al menos un incendio histórico registrado (top_celdas.parquet,
usado para entrenar y para el monitoreo en alerta_incendios.py).

El objetivo es identificar departamentos / zonas de la provincia que NO
están siendo monitoreadas porque nunca tuvieron un incendio en el
histórico (FIRED), independientemente de que TOP_N ya esté en None.

CUÁNDO CORRER ESTE SCRIPT:
 1. Primera vez (para generar grid_salta_completo.parquet)  ← ya hecho
 2. Después de reentrenar el modelo con datos nuevos
 No es necesario para corridas diarias del sistema de alerta.

"""

import time
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from pathlib import Path
from shapely.geometry import Point

# ========================= CONFIGURACIÓN =========================
ERA5_DIR  = Path("era5_salta")
GRID_RES  = 0.25  # resolución del grid ERA5 (grados), inferida de lat_era5/lon_era5
# bbox usado también en alerta_incendios.py para la consulta a FIRMS
# (lon_min, lat_min, lon_max, lat_max)
BBOX_SALTA = (-68.5, -26.5, -62.0, -21.5)
# =================================================================


def cargar_limite_salta():
    limite_path = ERA5_DIR / "salta_provincia.gpkg"
    if not limite_path.exists():
        print("Descargando límite de Salta desde OpenStreetMap...")
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


def generar_grid_era5(bbox, res=GRID_RES):
    """Genera todas las celdas ERA5 (lat/lon) que cubren el bbox dado."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lons = np.arange(np.floor(lon_min / res) * res, np.ceil(lon_max / res) * res + res, res)
    lats = np.arange(np.floor(lat_min / res) * res, np.ceil(lat_max / res) * res + res, res)
    lons = np.round(lons, 2)
    lats = np.round(lats, 2)
    grid = pd.DataFrame(
        [(la, lo) for la in lats for lo in lons],
        columns=['lat_era5', 'lon_era5']
    )
    return grid


def main():
    limite_salta = cargar_limite_salta()

    # ---- 1. Grid ERA5 completo, recortado al límite de Salta ----
    print("Generando grid ERA5 completo...")
    grid = generar_grid_era5(BBOX_SALTA)
    gdf_grid = gpd.GeoDataFrame(
        grid,
        geometry=[Point(lon, lat) for lat, lon in zip(grid['lat_era5'], grid['lon_era5'])],
        crs="EPSG:4326"
    )
    dentro = gpd.sjoin(gdf_grid, limite_salta[['geometry']], how='inner', predicate='within')
    grid_salta = grid.loc[dentro.index].copy().reset_index(drop=True)
    print(f"Celdas ERA5 totales en el bbox: {len(grid)}")
    print(f"Celdas ERA5 dentro del límite de Salta: {len(grid_salta)}")

    # ---- 2. Celdas con incendio histórico (las que sí se usan hoy) ----
    top_celdas_path = ERA5_DIR / "top_celdas.parquet"
    if not top_celdas_path.exists():
        raise FileNotFoundError(
            f"No se encontró {top_celdas_path}. Corré primero entrenamiento_xgb.py."
        )
    top_celdas = pd.read_parquet(top_celdas_path)
    print(f"Celdas con incendio histórico (dataset actual): {len(top_celdas)}")

    # ---- 3. Diferencia: celdas del grid SIN incendio histórico ----
    merged = grid_salta.merge(
        top_celdas[['lat_era5', 'lon_era5']],
        on=['lat_era5', 'lon_era5'],
        how='left',
        indicator=True
    )
    faltantes = merged[merged['_merge'] == 'left_only'][['lat_era5', 'lon_era5']].copy()
    print(f"\nCeldas dentro de Salta SIN incendio histórico registrado: {len(faltantes)}")
    print(f"  (cobertura actual: {len(top_celdas)}/{len(grid_salta)} = "
          f"{100*len(top_celdas)/len(grid_salta):.1f}% del grid provincial)")

    if faltantes.empty:
        print("\n✅ No hay celdas faltantes: el dataset ya cubre todo el grid de Salta.")
        return

    # ---- 4. Departamento de cada celda faltante (con cache compartido) ----
    depto_path = ERA5_DIR / "departamentos_celdas.parquet"
    if depto_path.exists():
        depto_cache = pd.read_parquet(depto_path)
        print(f"\nCache de departamentos encontrado: {len(depto_cache)} celdas")
    else:
        depto_cache = pd.DataFrame(columns=['lat_era5', 'lon_era5', 'departamento'])

    faltantes_con_cache = faltantes.merge(
        depto_cache, on=['lat_era5', 'lon_era5'], how='left'
    )
    faltantes_sin_depto = faltantes_con_cache[
        faltantes_con_cache['departamento'].isna()
    ][['lat_era5', 'lon_era5']]
    print(f"  Celdas nuevas a consultar (no estaban en cache): {len(faltantes_sin_depto)}")

    nuevos_deptos = []
    for _, row in faltantes_sin_depto.iterrows():
        depto = obtener_departamento(row['lat_era5'], row['lon_era5'])
        nuevos_deptos.append({
            'lat_era5': row['lat_era5'],
            'lon_era5': row['lon_era5'],
            'departamento': depto
        })
        time.sleep(0.2)

    if nuevos_deptos:
        depto_cache = pd.concat([depto_cache, pd.DataFrame(nuevos_deptos)], ignore_index=True)
        depto_cache = depto_cache.drop_duplicates(subset=['lat_era5', 'lon_era5'], keep='last')
        depto_cache.to_parquet(depto_path, index=False)
        print(f"  ✅ Cache de departamentos actualizado ({len(depto_cache)} celdas en total)")

    faltantes = faltantes.merge(depto_cache, on=['lat_era5', 'lon_era5'], how='left')

    # ---- 5. Resumen por departamento ----
    print("\n===== CELDAS SIN INCENDIO HISTÓRICO, POR DEPARTAMENTO =====")
    resumen = faltantes.groupby('departamento').size().sort_values(ascending=False)
    print(resumen.to_string())

    # ---- 6. También: departamentos cubiertos vs. no cubiertos en absoluto ----
    deptos_cubiertos = set(top_celdas['departamento'].unique())
    deptos_no_cubiertos = set(faltantes['departamento'].unique()) - deptos_cubiertos
    if deptos_no_cubiertos:
        print("\n⚠️ Departamentos que NO tienen NINGUNA celda con incendio histórico")
        print("   (es decir, hoy no están monitoreados en absoluto):")
        for d in sorted(deptos_no_cubiertos):
            n = (faltantes['departamento'] == d).sum()
            print(f"   - {d}: {n} celdas sin cubrir")
    else:
        print("\n✅ Todos los departamentos con celdas faltantes ya tienen al menos "
              "una celda cubierta en otro punto del departamento.")

    # ---- 7. Chequeo puntual: ¿está Cafayate cubierto? ----
    print("\n===== CHEQUEO PUNTUAL: CAFAYATE =====")
    deptos_en_grid = pd.concat([
        top_celdas[['lat_era5', 'lon_era5', 'departamento']],
        faltantes[['lat_era5', 'lon_era5', 'departamento']]
    ], ignore_index=True)
    cafayate_celdas = deptos_en_grid[
        deptos_en_grid['departamento'].str.contains('Cafayate', case=False, na=False)
    ]
    if cafayate_celdas.empty:
        print("⚠️ No se encontró ninguna celda con departamento 'Cafayate' en el grid generado. "
              "Revisar manualmente contra el límite provincial / la API de georef.")
    else:
        cubiertas = cafayate_celdas.merge(
            top_celdas[['lat_era5', 'lon_era5']], on=['lat_era5', 'lon_era5'], how='inner'
        )
        print(f"Celdas en el departamento Cafayate: {len(cafayate_celdas)}")
        print(f"  Con incendio histórico (monitoreadas hasta ahora): {len(cubiertas)}")
        print(f"  SIN incendio histórico (recién se van a empezar a monitorear): "
              f"{len(cafayate_celdas) - len(cubiertas)}")
        print(cafayate_celdas.to_string(index=False))

    # ---- 8. Altitud para las celdas faltantes (mismo cache que usa el entrenamiento) ----
    altitud_path = ERA5_DIR / "altitud_celdas.parquet"
    if altitud_path.exists():
        alt_cache = pd.read_parquet(altitud_path)
        print(f"\nCache de altitud encontrado: {len(alt_cache)} celdas")
    else:
        alt_cache = pd.DataFrame(columns=['lat_era5', 'lon_era5', 'altitud'])

    def obtener_altitud(lat, lon):
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()['elevation'][0]

    faltantes_alt_cache = faltantes[['lat_era5', 'lon_era5']].merge(
        alt_cache, on=['lat_era5', 'lon_era5'], how='left'
    )
    faltantes_sin_alt = faltantes_alt_cache[
        faltantes_alt_cache['altitud'].isna()
    ][['lat_era5', 'lon_era5']]
    print(f"  Celdas sin altitud en cache (a consultar): {len(faltantes_sin_alt)}")

    nuevas_altitudes = []
    for _, row in faltantes_sin_alt.iterrows():
        alt = obtener_altitud(row['lat_era5'], row['lon_era5'])
        nuevas_altitudes.append({'lat_era5': row['lat_era5'], 'lon_era5': row['lon_era5'], 'altitud': alt})
        time.sleep(0.1)

    if nuevas_altitudes:
        alt_cache = pd.concat([alt_cache, pd.DataFrame(nuevas_altitudes)], ignore_index=True)
        alt_cache = alt_cache.drop_duplicates(subset=['lat_era5', 'lon_era5'], keep='last')
        alt_cache.to_parquet(altitud_path, index=False)
        print(f"  ✅ Cache de altitud actualizado ({len(alt_cache)} celdas en total)")

    # ---- 9. Guardar detalle de celdas faltantes (diagnóstico) ----
    out_path = ERA5_DIR / "celdas_sin_incendio_historico.parquet"
    faltantes.to_parquet(out_path, index=False)
    print(f"\n✅ Detalle de celdas faltantes guardado en: {out_path}")

    # ---- 10. Construir y guardar el GRID COMPLETO DE MONITOREO ----
    # Este es el archivo que alerta_incendios.py debe usar en vez de top_celdas.parquet,
    # para que el sistema vigile TODA la provincia y no solo las celdas con historial.
    top_celdas_completo = top_celdas[['lat_era5', 'lon_era5', 'frecuencia', 'departamento']].copy()
    top_celdas_completo['tiene_historial_incendio'] = True

    faltantes_completo = faltantes[['lat_era5', 'lon_era5', 'departamento']].copy()
    faltantes_completo['frecuencia'] = 0
    faltantes_completo['tiene_historial_incendio'] = False

    grid_completo = pd.concat([top_celdas_completo, faltantes_completo], ignore_index=True)
    grid_completo = grid_completo.drop_duplicates(subset=['lat_era5', 'lon_era5'])

    grid_path = ERA5_DIR / "grid_salta_completo.parquet"
    grid_completo.to_parquet(grid_path, index=False)
    print(f"\n✅ Grid completo de monitoreo guardado: {grid_path}")
    print(f"   - Total celdas: {len(grid_completo)}")
    print(f"   - Con historial de incendio: {int(grid_completo['tiene_historial_incendio'].sum())}")
    print(f"   - Sin historial (recién incorporadas al monitoreo): "
          f"{int((~grid_completo['tiene_historial_incendio']).sum())}")
    print("\n⚠️ Para que esto tenga efecto, alerta_incendios.py debe leer "
          "grid_salta_completo.parquet en vez de top_celdas.parquet.")


if __name__ == "__main__":
    main()

