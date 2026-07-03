#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
alerta_incendios_v2.py
======================
Cambios respecto a la versión anterior:

  1. Balance hídrico usa t2m_max y t2m_mean reales de los últimos 7 días
     (API de archivo), no el valor del día objetivo como proxy para todos.
     Esto elimina la principal discrepancia train/inference en swvl1_aprox.

  2. Assert explícito de orden de features antes de llamar a predict_proba.
     Si el orden no coincide con el modelo entrenado, falla ruidosamente
     en lugar de producir probabilidades silenciosamente incorrectas.

  3. Al cargar el modelo se loguean los metadatos guardados en el pickle
     (versión, fecha de entrenamiento, n_celdas, umbral provisional).
     Permite verificar coherencia sin abrir el pickle manualmente.

  4. Si el modelo no trae calibrador (v2 en adelante) se maneja sin error.

Invariantes que se mantienen:
  - FULL_FEATURES y su orden deben ser idénticos a entrenamiento_xgb_v2.py.
    Cualquier cambio en uno debe replicarse en el otro.
  - USAR_CALIBRACION=False: las probabilidades calibradas no son confiables
    en producción por la diferencia de tasa base (ver comentario en sección
    de calibración del script de entrenamiento).
  - grid_salta_completo.parquet como fuente de celdas a monitorear (220).
    Fallback a top_celdas.parquet si no existe.
"""

import requests
import pandas as pd
import numpy as np
import pickle
import folium
import geopandas as gpd
import time
from shapely.geometry import Point
from sklearn.neighbors import BallTree
from datetime import datetime, timedelta
from pathlib import Path

# ========================= CONFIGURACIÓN =========================
ERA5_DIR        = Path("era5_salta")
TELEGRAM_TOKEN  = "8642047851:AAG5VzeZfNfEqhPmRv0JDFLuvERd8a2oJWU"
TELEGRAM_CHAT_ID = "642106059"
FIRMS_MAP_KEY   = "22773fe1b8d30f46d488fffcfd104aaa"
UMBRAL          = 0.35   # Ajustable. Ver barrido de umbrales en backtesting.
USAR_CALIBRACION = False  # Ver comentario en entrenamiento_xgb_v2.py sección 17.
# =================================================================


# ──────────────────────────────────────────────────────────────────
# Funciones auxiliares
# ──────────────────────────────────────────────────────────────────

def cargar_limite_salta():
    limite_path = ERA5_DIR / "salta_provincia.gpkg"
    if not limite_path.exists():
        print("Descargando límite de Salta desde OpenStreetMap...")
        import osmnx as ox
        salta = ox.geocode_to_gdf("Provincia de Salta, Argentina")
        salta = salta.to_crs("EPSG:4326")
        salta.to_file(limite_path, driver='GPKG')
        print("✅ Límite de Salta guardado.")
    return gpd.read_file(limite_path)


def filtrar_en_salta(df, limite_salta):
    if df.empty:
        return df
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(lon, lat) for lat, lon in zip(df['lon'], df['lat'])],
        crs="EPSG:4326"
    )
    dentro = gpd.sjoin(gdf, limite_salta[['geometry']], how='inner', predicate='within')
    return df.loc[dentro.index].copy()


def cargar_modelo():
    with open(ERA5_DIR / "modelo_xgb_mejorado.pkl", "rb") as f:
        data = pickle.load(f)

    # CAMBIO v2: loguear metadatos del pickle para verificar coherencia.
    version = data.get('version', 'v1 (sin metadatos)')
    print(f"  Versión del modelo:        {version}")
    print(f"  Fecha de entrenamiento:    {data.get('fecha_entrenamiento', 'desconocida')}")
    print(f"  Celdas de entrenamiento:   {data.get('n_celdas_entrenamiento', '?')}")
    print(f"  Train hasta año:           {data.get('train_hasta_anio', '?')}")
    print(f"  scale_pos_weight usado:    {data.get('scale_pos_weight_usado', '?')}")
    print(f"  Umbral provisional (ref):  {data.get('umbral_provisional', '?')}")
    print(f"  AUC test:                  {data.get('auc_test', '?')}")

    return data


def obtener_pronostico_dia_objetivo(lat, lon, fecha):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "relative_humidity_2m_min",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "wind_direction_10m_dominant",
            "precipitation_sum",
            "surface_pressure_min"
        ],
        "timezone":   "America/Argentina/Salta",
        "start_date": fecha.strftime("%Y-%m-%d"),
        "end_date":   fecha.strftime("%Y-%m-%d")
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def obtener_datos_historicos_7d(lat, lon, fecha_inicio, fecha_fin):
    """
    Descarga temperature_2m_max, temperature_2m_mean y precipitation_sum
    de los últimos 7 días desde la API de archivo de Open-Meteo.
    Devuelve un dict con listas diarias.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": fecha_inicio.strftime("%Y-%m-%d"),
        "end_date":   fecha_fin.strftime("%Y-%m-%d"),
        "daily":      "temperature_2m_max,temperature_2m_mean,precipitation_sum",
        "timezone":   "America/Argentina/Salta"
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()['daily']


def obtener_fuegos_activos(bbox_coords):
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{FIRMS_MAP_KEY}/VIIRS_NOAA20_NRT/"
        f"{bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]}/1"
    )
    try:
        df = pd.read_csv(url)
        print(f"✅ FIRMS: {len(df)} focos activos en las últimas 24h.")
        return df
    except Exception as e:
        print(f"⚠️ Error al descargar FIRMS: {e}. Se asume 0 focos activos.")
        return pd.DataFrame()


def obtener_presion_punto(lat, lon, fecha, nombre="punto", retries=3):
    """Obtiene presión mínima diaria en un punto geográfico con reintentos."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "daily":      "surface_pressure_min",
        "timezone":   "America/Argentina/Salta",
        "start_date": fecha.strftime("%Y-%m-%d"),
        "end_date":   fecha.strftime("%Y-%m-%d")
    }
    for i in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()['daily']['surface_pressure_min'][0]
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"  Reintentando presión {nombre} ({i+1}/{retries})...")
            time.sleep(2)


def asignar_fuegos_a_celdas(focos_df, celdas_era5):
    if focos_df.empty or 'latitude' not in focos_df.columns or celdas_era5.empty:
        return celdas_era5.assign(fuegos_activos_lag1=0)
    coords_celdas = np.radians(celdas_era5[['lat_era5', 'lon_era5']].values)
    coords_focos  = np.radians(focos_df[['latitude', 'longitude']].values)
    tree = BallTree(coords_celdas, metric='haversine')
    _, idx = tree.query(coords_focos, k=1)
    focos_df = focos_df.copy()
    focos_df['lat_era5'] = celdas_era5.iloc[idx.flatten()]['lat_era5'].values
    focos_df['lon_era5'] = celdas_era5.iloc[idx.flatten()]['lon_era5'].values
    conteo = (focos_df
              .groupby(['lat_era5', 'lon_era5'])
              .size()
              .reset_index(name='fuegos_activos_lag1'))
    resultado = celdas_era5.merge(conteo, on=['lat_era5', 'lon_era5'], how='left')
    resultado['fuegos_activos_lag1'] = resultado['fuegos_activos_lag1'].fillna(0).astype(int)
    return resultado


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
        cache[clave] = "Desconocido"
        return "Desconocido"
    except Exception:
        cache[clave] = "Error"
        return "Error"


# ──────────────────────────────────────────────────────────────────
# Mapa
# ──────────────────────────────────────────────────────────────────

def generar_mapa(alertas_df, limite_salta, fecha_alerta, focos_activos_df=None):
    """
    Genera el mapa interactivo con dos capas independientes:

    Capa 1 — Focos FIRMS activos HOY (siempre visible, incluso sin alertas):
      Círculos rojos sólidos con el número de focos detectados en la celda.
      Fuente: VIIRS NOAA-20, últimas 24h.

    Capa 2 — Alertas predictivas MAÑANA:
      Top 5: círculos numerados con color según probabilidad.
      Resto: íconos de fuego azul.

    focos_activos_df: DataFrame con columnas lat_era5, lon_era5, fuegos_activos_lag1
                      (el mismo que produce asignar_fuegos_a_celdas). Puede ser None
                      o vacío — en ese caso la capa de focos no se dibuja.
    """
    mapa = folium.Map(location=[-24.5, -65.5], zoom_start=7, tiles="CartoDB positron")
    folium.GeoJson(
        limite_salta.to_crs("EPSG:4326").__geo_interface__,
        style_function=lambda _: {'color': '#2c3e50', 'weight': 2, 'dashArray': '5,5'}
    ).add_to(mapa)

    titulo_html = f"""
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
         z-index: 1000; background-color: white; padding: 10px 20px;
         border: 2px solid #c0392b; border-radius: 8px;
         font-family: Arial; font-size: 14px; font-weight: bold;">
         🔥 RIESGO DE INCENDIO — SALTA — {fecha_alerta.strftime('%d/%m/%Y')}
    </div>"""
    mapa.get_root().html.add_child(folium.Element(titulo_html))

    # ── CAPA 1: focos FIRMS activos hoy ──────────────────────────────
    # Se dibuja siempre, antes que las alertas predictivas, para que
    # quede debajo visualmente cuando hay superposición espacial.
    n_focos_activos = 0
    focos_en_leyenda = []

    if focos_activos_df is not None and not focos_activos_df.empty:
        focos_con_fuego = focos_activos_df[focos_activos_df['fuegos_activos_lag1'] > 0].copy()
        n_focos_activos = len(focos_con_fuego)

        for _, row in focos_con_fuego.iterrows():
            lat = row['lat_era5']
            lon = row['lon_era5']
            n  = int(row['fuegos_activos_lag1'])
            depto = obtener_departamento(lat, lon)
            focos_en_leyenda.append({'depto': depto, 'lat': lat, 'lon': lon, 'n': n})

            # Tamaño del círculo escala levemente con la cantidad de focos
            radio = 8 + min(n * 2, 10)

            popup_html = f"""
            <div style="font-family:Arial; font-size:13px; min-width:180px">
                <b style="color:#c0392b">🔴 FOCO ACTIVO HOY (FIRMS)</b>
                <hr style="margin:5px 0">
                <b>Departamento:</b> {depto}<br>
                <b>Focos detectados:</b> {n}<br>
                <b>Coordenadas:</b> {abs(lat):.2f}°S, {abs(lon):.2f}°O<br>
                <b>Fuente:</b> VIIRS NOAA-20, últimas 24h<br>
                <i style="color:#777">Dato observado — no predictivo</i>
            </div>"""

            folium.CircleMarker(
                location=[lat, lon],
                radius=radio,
                color='#c0392b',
                fill=True,
                fill_color='#e74c3c',
                fill_opacity=0.85,
                weight=2,
                popup=folium.Popup(popup_html, max_width=240),
                tooltip=f"🔴 {depto} — {n} foco(s) hoy"
            ).add_to(mapa)

    # ── CAPA 2: alertas predictivas mañana ───────────────────────────
    if not alertas_df.empty:
        alertas_ordenadas = alertas_df.sort_values('probabilidad', ascending=False).reset_index(drop=True)
        alertas_ordenadas['ranking'] = alertas_ordenadas.index + 1
        top5  = alertas_ordenadas.head(5)
        resto = alertas_ordenadas.iloc[5:]

        for _, row in top5.iterrows():
            num = row['ranking']
            if row['probabilidad'] >= 0.50:
                bg_color = '#e74c3c'
            elif row['probabilidad'] >= 0.40:
                bg_color = '#e67e22'
            else:
                bg_color = '#f1c40f'

            icon = folium.DivIcon(
                html=f"""<div style="background-color:{bg_color}; border-radius:50%; width:28px;
                         height:28px; display:flex; align-items:center; justify-content:center;
                         font-weight:bold; color:white; border:2px solid black; font-size:14px;">
                         {num}</div>""",
                icon_size=(28, 28), icon_anchor=(14, 14)
            )
            depto = row.get('departamento', obtener_departamento(row['lat'], row['lon']))
            popup_html = f"""
            <div style="font-family:Arial; font-size:13px; min-width:200px">
                <b style="color:{bg_color}">🔥 ALERTA PREDICTIVA (Puesto {num})</b>
                <hr style="margin:5px 0">
                <b>Departamento:</b> {depto}<br>
                <b>Probabilidad mañana:</b> {row['probabilidad']:.1%}<br>
                <b>Coordenadas:</b> {abs(row['lat']):.2f}°S, {abs(row['lon']):.2f}°O<br>
                <b>Factores:</b> {row.get('explicacion', 'Consultar mensaje')}<br>
                <i style="color:#777">Predicción para {fecha_alerta.strftime('%d/%m/%Y')}</i>
            </div>"""
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"#{num} {depto} — {row['probabilidad']:.1%}",
                icon=icon
            ).add_to(mapa)

        for _, row in resto.iterrows():
            icon = folium.Icon(color='blue', icon='fire', prefix='fa')
            depto = row.get('departamento', obtener_departamento(row['lat'], row['lon']))
            popup_html = f"""
            <div style="font-family:Arial; font-size:13px; min-width:200px">
                <b style="color:#3498db">🔥 ALERTA PREDICTIVA (Puesto {row['ranking']})</b>
                <hr style="margin:5px 0">
                <b>Departamento:</b> {depto}<br>
                <b>Probabilidad mañana:</b> {row['probabilidad']:.1%}<br>
                <b>Coordenadas:</b> {abs(row['lat']):.2f}°S, {abs(row['lon']):.2f}°O<br>
                <b>Factores:</b> {row.get('explicacion', 'Consultar mensaje')}<br>
                <i style="color:#777">Predicción para {fecha_alerta.strftime('%d/%m/%Y')}</i>
            </div>"""
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"#{row['ranking']} {depto} — {row['probabilidad']:.1%}",
                icon=icon
            ).add_to(mapa)

    # ── LEYENDA ───────────────────────────────────────────────────────
    # Sección superior fija: referencias de capas.
    # Sección inferior scrollable: top 20 alertas predictivas.

    # Tabla focos activos (si los hay)
    if focos_en_leyenda:
        tabla_focos = '<table style="border-collapse:collapse; width:100%; font-size:12px; margin-bottom:4px;">'
        for f in focos_en_leyenda:
            tabla_focos += f"""
            <tr>
              <td style="padding:2px 4px;">
                <span style="display:inline-block; width:10px; height:10px;
                  background:#e74c3c; border-radius:50%; border:1px solid #c0392b;">
                </span>
              </td>
              <td style="padding:2px 6px">{f['depto']}</td>
              <td style="padding:2px 4px; color:#c0392b; font-weight:bold; text-align:right">
                {f['n']} foco(s)
              </td>
            </tr>"""
        tabla_focos += '</table>'
        seccion_focos = f"""
        <b style="font-size:12px; color:#c0392b">🔴 FOCOS ACTIVOS HOY (FIRMS)</b><br>
        <span style="font-size:10px; color:#777">VIIRS NOAA-20 · últimas 24h · dato observado</span><br>
        {tabla_focos}
        <hr style="margin:8px 0">"""
    else:
        seccion_focos = """
        <b style="font-size:12px; color:#27ae60">✅ Sin focos activos hoy (FIRMS)</b><br>
        <span style="font-size:10px; color:#777">VIIRS NOAA-20 · últimas 24h</span>
        <hr style="margin:8px 0">"""

    # Tabla top 20 alertas predictivas
    if not alertas_df.empty:
        alertas_ord2 = alertas_df.sort_values('probabilidad', ascending=False).reset_index(drop=True)
        alertas_ord2['ranking'] = alertas_ord2.index + 1
        top20 = alertas_ord2.head(20)
        tabla_alertas = '<div style="max-height:280px; overflow-y:auto;">'
        tabla_alertas += '<table style="border-collapse:collapse; width:100%; font-size:12px;">'
        for _, row in top20.iterrows():
            depto = row.get('departamento', obtener_departamento(row['lat'], row['lon']))
            color_fondo = '#e74c3c' if row['ranking'] <= 5 else '#3498db'
            tabla_alertas += f"""
            <tr>
              <td style="padding:2px 4px; font-weight:bold; color:white; background:{color_fondo};
                         border-radius:50%; text-align:center; width:24px">{row['ranking']}</td>
              <td style="padding:2px 8px">{depto}</td>
              <td style="padding:2px 4px; font-weight:bold; color:{color_fondo};
                         text-align:right">{row['probabilidad']:.1%}</td>
            </tr>"""
        tabla_alertas += '</table></div>'
        seccion_alertas = f"""
        <b style="font-size:12px">🔥 PREDICCIÓN MAÑANA — TOP 20</b><br>
        <span style="font-size:10px; color:#777">Probabilidad de incendio · {fecha_alerta.strftime('%d/%m/%Y')}</span><br>
        {tabla_alertas}
        <hr style="margin:8px 0">"""
    else:
        seccion_alertas = """
        <b style="font-size:12px; color:#27ae60">🟢 Sin alertas predictivas para mañana</b><br>
        <span style="font-size:10px; color:#777">Ninguna celda superó el umbral de alerta</span>
        <hr style="margin:8px 0">"""

    leyenda_html = f"""
    <div style="position: fixed; bottom: 20px; left: 15px; z-index: 1000;
         background-color: white; padding: 14px 16px; border: 1px solid #ccc;
         border-radius: 8px; font-family: Arial; font-size: 12px; min-width: 290px;
         max-width: 320px; max-height: 75vh; overflow-y: auto;
         box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">

      {seccion_focos}
      {seccion_alertas}

      <b style="font-size:11px; color:#555">REFERENCIAS</b><br>
      <span style="color:#e74c3c;">●</span> Foco activo FIRMS (hoy)<br>
      <span style="color:#e74c3c; font-weight:bold">①②③</span> Top 5 alertas predictivas<br>
      <span style="color:#3498db;">🔥</span> Otras alertas predictivas (6-20)<br>
      <hr style="margin:6px 0">
      <span style="font-size:10px; color:#777">
        Hacé click en cualquier punto para ver detalles.<br>
        ╌ Límite provincial (OpenStreetMap)
      </span>
    </div>"""
    mapa.get_root().html.add_child(folium.Element(leyenda_html))

    mapa_path = ERA5_DIR / f"mapa_riesgo_{fecha_alerta}.html"
    mapa.save(str(mapa_path))
    n_alertas = len(alertas_df) if not alertas_df.empty else 0
    print(f"🗺 Mapa guardado: {mapa_path} "
          f"({n_alertas} alertas predictivas, {n_focos_activos} focos FIRMS activos)")
    return mapa_path


def enviar_telegram_con_mapa(mensaje_texto, archivo_mapa=None):
    url_text = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url_text,
                          json={"chat_id": TELEGRAM_CHAT_ID,
                                "text": mensaje_texto,
                                "parse_mode": "HTML"},
                          timeout=10)
        r.raise_for_status()
        print("✅ Mensaje enviado a Telegram")
    except Exception as e:
        print(f"❌ Error al enviar mensaje: {e}")

    if archivo_mapa and Path(archivo_mapa).exists():
        url_doc = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
        try:
            with open(archivo_mapa, 'rb') as f:
                r = requests.post(
                    url_doc,
                    files={'document': f},
                    data={'chat_id': TELEGRAM_CHAT_ID,
                          'caption': '🗺 Mapa interactivo — la leyenda muestra el Top 20'},
                    timeout=30
                )
            r.raise_for_status()
            print("✅ Mapa enviado a Telegram")
        except Exception as e:
            print(f"❌ Error al enviar mapa: {e}")


def explicar_riesgo(row):
    factores = []
    if row['t2m_max'] > 35:
        factores.append(f"temperatura extremadamente alta ({row['t2m_max']:.0f}°C)")
    elif row['t2m_max'] > 30:
        factores.append(f"temperatura muy alta ({row['t2m_max']:.0f}°C)")
    elif row['t2m_max'] > 28:
        factores.append(f"temperatura elevada ({row['t2m_max']:.0f}°C)")
    if row['dias_sin_lluvia'] > 15:
        factores.append(f"sequía extrema ({row['dias_sin_lluvia']} días sin lluvia)")
    elif row['dias_sin_lluvia'] > 10:
        factores.append(f"sequía prolongada ({row['dias_sin_lluvia']} días sin lluvia)")
    elif row['dias_sin_lluvia'] > 5:
        factores.append(f"varios días sin lluvia ({row['dias_sin_lluvia']:.0f})")
    elif row['dias_sin_lluvia'] > 2:
        factores.append(f"sin lluvia en los últimos {row['dias_sin_lluvia']:.0f} días")
    if row['wind_max'] > 25:
        factores.append(f"viento muy fuerte ({row['wind_max']:.0f} km/h)")
    elif row['wind_max'] > 15:
        factores.append(f"viento fuerte ({row['wind_max']:.0f} km/h)")
    elif row['wind_max'] > 10:
        factores.append(f"viento moderado ({row['wind_max']:.0f} km/h)")
    if row['swvl1_aprox'] < 0.12:
        factores.append("suelo extremadamente seco")
    elif row['swvl1_aprox'] < 0.15:
        factores.append("suelo seco")
    if row['fuegos_activos_lag1'] > 0:
        factores.append(f"ayer hubo {row['fuegos_activos_lag1']} foco activo en la zona")
    if row['es_feriado'] == 1:
        factores.append("día feriado (mayor actividad humana)")
    elif row['es_finde_semana'] == 1:
        factores.append("fin de semana (mayor actividad humana)")
    if row['ndvi_anomalia_aprox'] < -0.15:
        factores.append("vegetación mucho más seca de lo normal")
    elif row['ndvi_anomalia_aprox'] < -0.08:
        factores.append("vegetación más seca de lo normal")
    if row['tasa_rayos'] > 0.3:
        factores.append("alta probabilidad de rayos")
    elif row['tasa_rayos'] > 0.15:
        factores.append("probabilidad moderada de rayos")
    if row['densidad_poblacional'] > 50:
        factores.append("zona muy poblada")
    elif row['densidad_poblacional'] > 20:
        factores.append("zona poblada")
    if 'hr_mean' in row and row['hr_mean'] < 30:
        factores.append(f"humedad muy baja ({row['hr_mean']:.0f}%)")
    elif 'hr_mean' in row and row['hr_mean'] < 40:
        factores.append(f"humedad baja ({row['hr_mean']:.0f}%)")
    if 'fwi' in row and row['fwi'] > 50:
        factores.append(f"índice de peligro extremo (FWI={row['fwi']:.0f})")
    elif 'fwi' in row and row['fwi'] > 30:
        factores.append(f"índice de peligro alto (FWI={row['fwi']:.0f})")
    if not factores:
        return "Condiciones dentro de parámetros normales, pero se recomienda vigilancia."
    texto = ", ".join(factores)
    return texto[0].upper() + texto[1:] + "."


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    # ---- Cargar modelo ----
    print("Cargando modelo...")
    model_data   = cargar_modelo()
    model        = model_data['modelo']
    features     = model_data['features']
    calibrador   = model_data.get('calibrador')   # None en modelos v2+

    if calibrador is not None and USAR_CALIBRACION:
        print("✅ Calibrador encontrado y activado.")
    elif calibrador is not None and not USAR_CALIBRACION:
        print("ℹ️ Calibrador encontrado pero USAR_CALIBRACION=False → probabilidades crudas.")
    else:
        print("ℹ️ Modelo sin calibrador (v2+) → probabilidades crudas.")

    limite_salta = cargar_limite_salta()

    # ---- Tablas auxiliares ----
    clim_t2m  = pd.read_parquet(ERA5_DIR / "climatologia_t2m.parquet")
    clim_ndvi = pd.read_parquet(ERA5_DIR / "climatologia_ndvi.parquet")
    dens_pob  = pd.read_parquet(ERA5_DIR / "densidad_poblacional_celdas.parquet")
    rayos_df  = pd.read_parquet(ERA5_DIR / "rayos_celdas.parquet")

    # ---- Celdas a monitorear ----
    grid_path = ERA5_DIR / "grid_salta_completo.parquet"
    if grid_path.exists():
        celdas_grid = pd.read_parquet(grid_path)
    else:
        print("⚠️ No se encontró grid_salta_completo.parquet. Usando top_celdas.parquet.")
        celdas_grid = pd.read_parquet(ERA5_DIR / "top_celdas.parquet")

    alt_celdas = pd.read_parquet(ERA5_DIR / "altitud_celdas.parquet")
    alt_dict   = alt_celdas.set_index(['lat_era5', 'lon_era5'])['altitud'].to_dict()

    fwi_clim_path = ERA5_DIR / "fwi_climatologia.parquet"
    fwi_clim = pd.read_parquet(fwi_clim_path) if fwi_clim_path.exists() else None
    if fwi_clim is None:
        print("⚠️ No se encontró fwi_climatologia.parquet. FWI/ISI/BUI = 0.")

    # Las celdas de grid_salta_completo.parquet ya están dentro del límite provincial
    # por construcción (chequeo_cobertura.py las filtró con el mismo sjoin).
    # No se vuelve a filtrar aquí para evitar falsos vacíos por diferencias de CRS.
    focos     = celdas_grid[['lat_era5', 'lon_era5']].copy()
    depto_map = celdas_grid.set_index(['lat_era5', 'lon_era5'])['departamento'].to_dict()
    print(f"Monitoreando {len(focos)} celdas (grid provincial completo)")
    if len(focos) == 0:
        raise RuntimeError(
            "❌ grid_salta_completo.parquet está vacío o no tiene columnas lat_era5/lon_era5. "
            "Correr chequeo_cobertura.py para regenerarlo."
        )

    # ---- Último NDVI observado ----
    fired    = pd.read_parquet(ERA5_DIR / "fired_completo.parquet")
    fired_h  = fired[['lat_era5', 'lon_era5', 'date', 'ndvi']].rename(columns={'date': 'fecha'})
    neg_hist = pd.read_parquet(ERA5_DIR / "negativos.parquet")[['lat_era5', 'lon_era5', 'fecha', 'ndvi']]
    df_hist  = pd.concat([fired_h, neg_hist], ignore_index=True)
    df_hist['fecha'] = pd.to_datetime(df_hist['fecha'])
    fecha_hoy = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    ultimo_ndvi = (
        df_hist[df_hist['fecha'] < fecha_hoy]
        .groupby(['lat_era5', 'lon_era5'])
        .last()['ndvi']
        .reset_index()
        .rename(columns={'ndvi': 'ndvi_obs'})
    )
    focos = focos.merge(ultimo_ndvi, on=['lat_era5', 'lon_era5'], how='left')
    focos['ndvi_obs'] = focos['ndvi_obs'].fillna(0.3)

    # ---- Densidad poblacional ----
    focos = focos.merge(dens_pob, on=['lat_era5', 'lon_era5'], how='left')
    focos['densidad_poblacional'] = focos['densidad_poblacional'].fillna(0)

    # ---- FIRMS ----
    bbox_salta    = [-68.5, -26.5, -62.0, -21.5]
    focos_activos = obtener_fuegos_activos(bbox_salta)
    celdas_era5   = focos[['lat_era5', 'lon_era5']].drop_duplicates()
    fuegos_x_celda = asignar_fuegos_a_celdas(focos_activos, celdas_era5)
    focos = focos.merge(fuegos_x_celda, on=['lat_era5', 'lon_era5'], how='left')
    focos['fuegos_activos_lag1'] = focos['fuegos_activos_lag1'].fillna(0).astype(int)
    print(f"Celdas con focos activos hoy: {(focos['fuegos_activos_lag1'] > 0).sum()}")

    # ---- Guardar focos FIRMS en historial ----
    fecha_hoy_str = datetime.now().strftime('%Y-%m-%d')
    historial_dir = ERA5_DIR / "historial"
    historial_dir.mkdir(exist_ok=True)
    firms_path = historial_dir / "historial_firms.csv"
    celdas_con_fuego = fuegos_x_celda[fuegos_x_celda['fuegos_activos_lag1'] > 0].copy()
    if not celdas_con_fuego.empty:
        celdas_con_fuego = celdas_con_fuego.rename(
            columns={'lat_era5': 'lat', 'lon_era5': 'lon', 'fuegos_activos_lag1': 'focos'}
        )
        celdas_con_fuego['fecha'] = fecha_hoy_str
        celdas_con_fuego = celdas_con_fuego[['fecha', 'lat', 'lon', 'focos']]
    else:
        celdas_con_fuego = pd.DataFrame([{
            'fecha': fecha_hoy_str, 'lat': None, 'lon': None, 'focos': 0
        }])
    if firms_path.exists():
        firms_hist = pd.read_csv(firms_path)
        firms_hist = firms_hist[firms_hist['fecha'] != fecha_hoy_str]
        firms_hist = pd.concat([firms_hist, celdas_con_fuego], ignore_index=True)
    else:
        firms_hist = celdas_con_fuego
    firms_hist.to_csv(firms_path, index=False)
    print(f"📋 Focos FIRMS guardados: {fecha_hoy_str}")

    # ---- Fecha pronóstico ----
    fecha_alerta = datetime.now().date() + timedelta(days=1)
    mes_alerta   = fecha_alerta.month
    dia_año      = fecha_alerta.timetuple().tm_yday
    print(f"Consultando pronóstico para {fecha_alerta}...")

    # ---- Feriados ----
    anio_alerta = fecha_alerta.year
    try:
        resp = requests.get(
            f"https://date.nager.at/api/v3/PublicHolidays/{anio_alerta}/AR", timeout=10
        )
        resp.raise_for_status()
        holidays_set = set(h['date'] for h in resp.json())
    except Exception:
        print("⚠️ No se pudieron obtener feriados.")
        holidays_set = set()

    # ---- Presión Chile y cordillera (una sola vez, fuera del bucle) ----
    print("Obteniendo presión en Chile y cordillera...")
    try:
        presion_chile_hpa = obtener_presion_punto(-23.65, -70.40, fecha_alerta, "Chile")
    except Exception as e:
        print(f"⚠️ Error presión Chile: {e}. Se usará 0.")
        presion_chile_hpa = 0
    try:
        presion_cordillera_hpa = obtener_presion_punto(-23.0, -64.5, fecha_alerta, "cordillera")
    except Exception as e:
        print(f"⚠️ Error presión cordillera: {e}. Se usará 0.")
        presion_cordillera_hpa = 0

    # ---- Procesar cada celda ----
    todos = []
    total = len(focos)

    for i, (_, row) in enumerate(focos.iterrows()):
        if i % 20 == 0:
            print(f"  Procesando... {i}/{total}")

        # Pronóstico del día objetivo
        try:
            data_fc = obtener_pronostico_dia_objetivo(row['lat_era5'], row['lon_era5'], fecha_alerta)
            celda   = data_fc['daily']
        except Exception as e:
            print(f"  ⚠️ Error pronóstico ({row['lat_era5']}, {row['lon_era5']}): {e}")
            continue

        t2m_max  = celda['temperature_2m_max'][0]
        t2m_mean = celda['temperature_2m_mean'][0]
        hr_mean  = celda['relative_humidity_2m_mean'][0]
        hr_min   = celda['relative_humidity_2m_min'][0]
        wind_max = celda['wind_speed_10m_max'][0] / 3.6
        wind_dir = celda['wind_direction_10m_dominant'][0]
        wind_gust = celda['wind_gusts_10m_max'][0] / 3.6
        precip   = celda['precipitation_sum'][0]
        sp_min   = celda['surface_pressure_min'][0] * 100

        # ── CAMBIO v2: balance hídrico con datos históricos reales de 7 días ──
        # En la versión anterior t2m_mean_7d y t2m_max_7d eran el valor del día
        # objetivo repetido 7 veces, sobreestimando o subestimando ET según la época.
        # Ahora se descargan los 7 días anteriores desde la API de archivo.
        fecha_7d_inicio = fecha_alerta - timedelta(days=7)
        fecha_7d_fin    = fecha_alerta - timedelta(days=1)
        try:
            hist_7d    = obtener_datos_historicos_7d(
                row['lat_era5'], row['lon_era5'], fecha_7d_inicio, fecha_7d_fin
            )
            t2m_max_hist  = [v if v is not None else t2m_max  for v in hist_7d['temperature_2m_max']]
            t2m_mean_hist = [v if v is not None else t2m_mean for v in hist_7d['temperature_2m_mean']]
            precip_hist   = [v if v is not None else 0.0      for v in hist_7d['precipitation_sum']]
        except Exception as e:
            print(f"  ⚠️ Error hist 7d ({row['lat_era5']}, {row['lon_era5']}): {e}. Usando día actual.")
            t2m_max_hist  = [t2m_max]  * 7
            t2m_mean_hist = [t2m_mean] * 7
            precip_hist   = [0.0]      * 7

        # Completar con el día objetivo para tener 7+1 días y calcular ventanas
        t2m_max_7d_val  = np.mean(t2m_max_hist)
        t2m_mean_7d_val = np.mean(t2m_mean_hist)
        precip_7d       = sum(precip_hist)

        et_7d = (0.0023
                 * (t2m_mean_7d_val + 17.8)
                 * max(t2m_max_7d_val - t2m_mean_7d_val, 0)
                 * 7)
        balance_hidrico_7d = precip_7d - et_7d
        swvl1_aprox = float(np.clip(0.1 + 0.005 * balance_hidrico_7d, 0.05, 0.35))

        # Días sin lluvia (desde el historial real de 7 días)
        dias_sin_lluvia = 0
        for p in reversed(precip_hist):
            if p <= 0.1:
                dias_sin_lluvia += 1
            else:
                break

        # Ventana de 3 días para t2m_max_3d_avg y precip_3d_sum
        # (los últimos 2 días históricos + el día objetivo)
        t2m_max_3d_avg = np.mean(t2m_max_hist[-2:] + [t2m_max])
        precip_3d_sum  = sum(precip_hist[-2:]) + precip
        # ── fin CAMBIO v2 ──────────────────────────────────────────────────

        # Gradientes de presión
        pressure_gradient      = (presion_chile_hpa * 100) - sp_min
        pressure_gradient_zonda = (presion_chile_hpa - presion_cordillera_hpa) * 100
        altitud = alt_dict.get((row['lat_era5'], row['lon_era5']), 1000)

        # Anomalías climatológicas
        norm_t = clim_t2m[
            (clim_t2m['lat_era5'] == row['lat_era5'])
            & (clim_t2m['lon_era5'] == row['lon_era5'])
            & (clim_t2m['mes']     == mes_alerta)
        ]
        t2m_clim   = norm_t['t2m_clim'].values[0] if len(norm_t) > 0 else t2m_mean
        t2m_anomalia = t2m_mean - t2m_clim

        norm_n = clim_ndvi[
            (clim_ndvi['lat_era5'] == row['lat_era5'])
            & (clim_ndvi['lon_era5'] == row['lon_era5'])
            & (clim_ndvi['mes']     == mes_alerta)
        ]
        ndvi_clim    = norm_n['ndvi_clim'].values[0] if len(norm_n) > 0 else 0.3
        ndvi_anomalia = row['ndvi_obs'] - ndvi_clim

        # Rayos
        mask_rayos = (
            (rayos_df['lat_era5'] == row['lat_era5'])
            & (rayos_df['lon_era5'] == row['lon_era5'])
            & (rayos_df['dia_año'] == dia_año)
        )
        tasa_rayos_vals = rayos_df.loc[mask_rayos, 'tasa_rayos'].values
        tasa_rayos = tasa_rayos_vals[0] if len(tasa_rayos_vals) > 0 else 0

        es_feriado = 1 if fecha_alerta.strftime('%Y-%m-%d') in holidays_set else 0
        es_finde   = 1 if fecha_alerta.weekday() >= 5 else 0

        rayos_ndvi  = tasa_rayos * max(ndvi_anomalia, 0)
        temp_ndvi   = t2m_anomalia * max(ndvi_anomalia, 0)
        fire_danger = max(t2m_max - 20, 0) * (wind_max / 10) * (1 - swvl1_aprox)
        temp_hr     = t2m_max * (100 - hr_mean) / 100

        if fwi_clim is not None:
            fwi_row = fwi_clim[
                (fwi_clim['lat_era5'] == row['lat_era5'])
                & (fwi_clim['lon_era5'] == row['lon_era5'])
                & (fwi_clim['mes']     == mes_alerta)
            ]
            if len(fwi_row) > 0:
                fwi_val = fwi_row.iloc[0]['fwi']
                isi_val = fwi_row.iloc[0]['isi']
                bui_val = fwi_row.iloc[0]['bui']
            else:
                fwi_val = isi_val = bui_val = 0
        else:
            fwi_val = isi_val = bui_val = 0

        todos.append({
            'lat': row['lat_era5'],
            'lon': row['lon_era5'],
            't2m_max':              t2m_max,
            't2m_mean':             t2m_mean,
            't2m_anomalia_aprox':   t2m_anomalia,
            'wind_max':             wind_max,
            'wind_dir':             wind_dir,
            'precip':               precip,
            'dias_sin_lluvia':      dias_sin_lluvia,
            'sp_min':               sp_min,
            'swvl1_aprox':          swvl1_aprox,
            'ndvi_lag15':           row['ndvi_obs'],
            'ndvi_anomalia_aprox':  ndvi_anomalia,
            'sin_dia':              np.sin(2 * np.pi * dia_año / 365.25),
            'cos_dia':              np.cos(2 * np.pi * dia_año / 365.25),
            'densidad_poblacional': row['densidad_poblacional'],
            'es_feriado':           es_feriado,
            'es_finde_semana':      es_finde,
            'tasa_rayos':           tasa_rayos,
            't2m_max_3d_avg':       t2m_max_3d_avg,
            'precip_3d_sum':        precip_3d_sum,
            'rayos_ndvi':           rayos_ndvi,
            'temp_ndvi':            temp_ndvi,
            'fire_danger':          fire_danger,
            'fuegos_activos_lag1':  row['fuegos_activos_lag1'],
            'hr_mean':              hr_mean,
            'temp_hr':              temp_hr,
            'fwi':                  fwi_val,
            'isi':                  isi_val,
            'bui':                  bui_val,
            'departamento':         depto_map.get((row['lat_era5'], row['lon_era5']), "Desconocido"),
            'pressure_gradient':            pressure_gradient,
            'wind_gust_max_chile':          wind_gust,
            'hr_min_chile':                 hr_min,
            'pressure_gradient_zonda':      pressure_gradient_zonda,
            'altitud':                      altitud,
        })

    df = pd.DataFrame(todos)

    # Rellenar features faltantes
    missing = set(features) - set(df.columns)
    if missing:
        print(f"⚠️ Features faltantes: {missing}, se rellenan con 0")
        for col in missing:
            df[col] = 0

    # CAMBIO v2: assert explícito de orden de features.
    # Si hay una discrepancia entre este script y el pickle, falla aquí
    # en lugar de producir probabilidades silenciosamente incorrectas.
    X = df[features]
    assert list(X.columns) == features, (
        "❌ El orden de features no coincide con el modelo entrenado. "
        "Verificar que FULL_FEATURES en este script sea idéntico al del pickle."
    )

    probas_raw = model.predict_proba(X)[:, 1]

    if calibrador is not None and USAR_CALIBRACION:
        probas = calibrador.predict_proba(probas_raw.reshape(-1, 1))[:, 1]
    else:
        probas = probas_raw

    # ---- Diagnóstico ----
    print("\n===== DIAGNÓSTICO DE FEATURES =====")
    print(f"Focos activos detectados: {df['fuegos_activos_lag1'].sum()}")
    celdas_con_fuego_diag = df[df['fuegos_activos_lag1'] > 0]
    if len(celdas_con_fuego_diag) > 0:
        print("Celdas con foco(s) activo(s) (FIRMS, últimas 24h):")
        for _, r in celdas_con_fuego_diag.iterrows():
            depto    = depto_map.get((r['lat'], r['lon']), "Desconocido")
            idx_prob = probas[df.index.get_loc(r.name)]
            print(f"  ({r['lat']:.2f}, {r['lon']:.2f}) — {depto} — "
                  f"{int(r['fuegos_activos_lag1'])} foco(s) — prob mañana: {idx_prob:.3f}")

    for col in ['t2m_max', 'wind_max', 'dias_sin_lluvia', 'swvl1_aprox',
                'fwi', 'isi', 'bui', 'pressure_gradient', 'pressure_gradient_zonda',
                'hr_min_chile', 'wind_gust_max_chile']:
        if col in df.columns:
            print(f"  {col}: min={df[col].min():.2f}  mean={df[col].mean():.2f}  max={df[col].max():.2f}")

    print("\n===== DIAGNÓSTICO DE PROBABILIDADES =====")
    print(f"Calibración aplicada: {'sí' if (calibrador is not None and USAR_CALIBRACION) else 'no'}")
    print(f"Prob mínima:  {probas_raw.min():.4f}")
    print(f"Prob media:   {probas_raw.mean():.4f}")
    print(f"Prob máxima:  {probas_raw.max():.4f}")
    print("\nTop 10 probabilidades:")
    top_probs = pd.DataFrame({'lat': df['lat'], 'lon': df['lon'], 'prob': probas})
    print(top_probs.sort_values('prob', ascending=False).head(10).to_string(index=False))
    print("\nAlertas potenciales por umbral:")
    for u in [0.20, 0.30, 0.35, 0.40, 0.45, 0.50]:
        print(f"  {u:.2f}: {(probas >= u).sum()} celdas")
    print("=========================================\n")

    df['probabilidad'] = probas
    df['alerta']       = (probas >= UMBRAL).astype(int)
    alertas            = df[df['alerta'] == 1].copy()
    alertas_salta      = filtrar_en_salta(alertas, limite_salta) if len(alertas) > 0 else alertas.copy()
    print(f"🔔 Alertas para {fecha_alerta}: {len(alertas_salta)} celdas")

    # ---- Guardar historial de alertas ----
    fecha_str    = fecha_alerta.strftime('%Y-%m-%d')
    path_alertas = historial_dir / "historial_alertas.csv"
    if len(alertas_salta) > 0:
        alertas_salta['fecha_pronostico'] = fecha_str
        cols_guardar = ['fecha_pronostico', 'lat', 'lon', 'probabilidad', 'departamento']
        if path_alertas.exists():
            old = pd.read_csv(path_alertas)
            new = pd.concat([old, alertas_salta[cols_guardar]], ignore_index=True)
        else:
            new = alertas_salta[cols_guardar]
        new.to_csv(path_alertas, index=False)
        print(f"💾 Historial de alertas actualizado: {len(alertas_salta)} nuevas alertas")

    # ---- Mensaje Telegram ----
    celdas_con_fuego_hoy = df[df['fuegos_activos_lag1'] > 0].copy()
    if len(celdas_con_fuego_hoy) > 0:
        focos_lines = ""
        for _, r in celdas_con_fuego_hoy.iterrows():
            depto = depto_map.get((r['lat'], r['lon']), "Desconocido")
            focos_lines += (f"  • {depto} ({abs(r['lat']):.2f}°S, {abs(r['lon']):.2f}°O) — "
                            f"{int(r['fuegos_activos_lag1'])} foco(s)\n")
        seccion_focos = (f"🔴 <b>Focos activos detectados hoy (FIRMS):</b>\n{focos_lines}"
                         f"<i>Fuente: VIIRS NOAA-20, últimas 24h.</i>\n\n")
    else:
        seccion_focos = "✅ <b>Sin focos activos detectados hoy (FIRMS).</b>\n\n"

    if len(alertas_salta) > 0:
        prob_prom = alertas_salta['probabilidad'].mean()
        top20     = alertas_salta.nlargest(20, 'probabilidad').reset_index(drop=True)
        top20_text = ""
        for i, row in top20.iterrows():
            if i < 5:
                explicacion = explicar_riesgo(row)
                top20_text += (f"  {i+1}. {row['departamento']} "
                               f"({abs(row['lat']):.2f}°S, {abs(row['lon']):.2f}°O) — "
                               f"{row['probabilidad']:.1%}\n  {explicacion}\n")
            else:
                top20_text += (f"  {i+1}. {row['departamento']} "
                               f"({abs(row['lat']):.2f}°S, {abs(row['lon']):.2f}°O) — "
                               f"{row['probabilidad']:.1%}\n")
        mensaje = (
            f"<b>🔥 ALERTA DE INCENDIOS FORESTALES</b>\n\n"
            f"📅 <b>Pronóstico:</b> {fecha_alerta.strftime('%d/%m/%Y')}\n\n"
            f"{seccion_focos}"
            f"📍 <b>Zonas en riesgo (mañana):</b> {len(alertas_salta)} de {len(focos)}\n"
            f"📊 <b>Prob. promedio:</b> {prob_prom:.1%}\n\n"
            f"<b>Top 20 zonas críticas</b> (ver mapa):\n{top20_text}\n"
            f"<i>Los números ① a ⑤ en el mapa corresponden al Top 5.</i>"
        )
        alertas_salta['explicacion'] = alertas_salta.apply(explicar_riesgo, axis=1)
        mapa_path = generar_mapa(alertas_salta, limite_salta, fecha_alerta,
                                 focos_activos_df=fuegos_x_celda)
    else:
        mensaje = (
            f"<b>📋 INFORME DE RIESGO DE INCENDIOS</b>\n\n"
            f"📅 <b>Pronóstico:</b> {fecha_alerta.strftime('%d/%m/%Y')}\n\n"
            f"{seccion_focos}"
            f"🟢 <b>Riesgo de propagación previsto para mañana: bajo en toda la provincia.</b>\n"
            f"<i>Ninguna celda superó el umbral de alerta ({UMBRAL:.0%}). "
            f"El monitoreo continúa.</i>"
        )
        mapa_path = generar_mapa(pd.DataFrame(), limite_salta, fecha_alerta,
                                 focos_activos_df=fuegos_x_celda)

    enviar_telegram_con_mapa(mensaje, mapa_path)

    # ---- Log de ejecución ----
    resumen = {
        'focos_activos_detectados': int(df['fuegos_activos_lag1'].sum()),
        'probabilidad_minima':      float(probas.min()),
        'probabilidad_media':       float(probas.mean()),
        'probabilidad_maxima':      float(probas.max()),
        'alertas_generadas':        len(alertas_salta),
    }
    log_path = historial_dir / "historial_ejecuciones.csv"
    log_row  = pd.DataFrame([{
        'fecha_ejecucion':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'fecha_pronostico':   str(fecha_alerta),
        'celdas_monitoreadas': len(focos),
        'focos_activos_hoy':  resumen['focos_activos_detectados'],
        'prob_max':           round(resumen['probabilidad_maxima'], 4),
        'prob_media':         round(resumen['probabilidad_media'],  4),
        'alertas_generadas':  resumen['alertas_generadas'],
        'umbral_usado':       UMBRAL,
        'version_modelo':     model_data.get('version', 'v1'),
    }])
    if log_path.exists():
        log_row.to_csv(log_path, mode='a', header=False, index=False)
    else:
        log_row.to_csv(log_path, index=False)
    print(f"📋 Log de ejecución actualizado: {log_path}")
    print("\n✅ Proceso completado.")
    return resumen


if __name__ == "__main__":
    resumen = main()
    print("Focos activos detectados:", resumen['focos_activos_detectados'])
    print("Probabilidad mínima:",      resumen['probabilidad_minima'])
    print("Probabilidad media:",       resumen['probabilidad_media'])
    print("Probabilidad máxima:",      resumen['probabilidad_maxima'])
    print("Alertas generadas:",        resumen['alertas_generadas'])

