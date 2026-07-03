#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
chequeo_diario.py
=================
Verifica si la predicción emitida AYER por alerta_incendios_v2.py
se cumplió HOY según los focos FIRMS registrados.

Corre después de alerta_incendios_v2.py (que ya guardó los focos
de hoy en historial_firms.csv antes de predecir para mañana).

Flujo:
  1. Lee la predicción de ayer desde historial_alertas.csv
     (fecha_pronostico == hoy).
  2. Lee los focos reales de hoy desde historial_firms.csv
     (fecha == hoy).
  3. Calcula VP / FP / FN / VN para ese único día.
  4. Guarda una línea en historial_chequeos_diarios.csv.
  5. Envía resumen a Telegram.

historial_chequeos_diarios.csv puede ser leído por evaluacion_v2.py
en el futuro para reconstruir métricas acumuladas sin recalcular
desde cero cada vez.
"""

import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import defaultdict

ERA5_DIR         = Path("era5_salta")
TELEGRAM_TOKEN   = "8642047851:AAG5VzeZfNfEqhPmRv0JDFLuvERd8a2oJWU"
TELEGRAM_CHAT_ID = "642106059"


# ──────────────────────────────────────────────────────────────────
def enviar_telegram(mensaje):
    url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": mensaje, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        print("✅ Mensaje Telegram enviado")
    except Exception as e:
        print(f"❌ Error al enviar Telegram: {e}")


# ──────────────────────────────────────────────────────────────────
def cargar_depto_map():
    grid_path = ERA5_DIR / "grid_salta_completo.parquet"
    if grid_path.exists():
        grid = pd.read_parquet(grid_path)
        return grid.set_index(['lat_era5', 'lon_era5'])['departamento'].to_dict()
    top = pd.read_parquet(ERA5_DIR / "top_celdas.parquet")
    return top.set_index(['lat_era5', 'lon_era5'])['departamento'].to_dict()


# ──────────────────────────────────────────────────────────────────
def chequear(fecha_verificar: date = None):
    """
    fecha_verificar: fecha cuya predicción se verifica (default: hoy).
    La predicción para ese día fue emitida ayer.
    """
    if fecha_verificar is None:
        fecha_verificar = datetime.now().date()

    fecha_str = fecha_verificar.strftime('%Y-%m-%d')
    print(f"\nVerificando predicción para {fecha_str}...")

    historial_dir = ERA5_DIR / "historial"
    depto_map     = cargar_depto_map()

    # ── 1. Predicción emitida ayer para hoy ───────────────────────
    path_alertas = historial_dir / "historial_alertas.csv"
    alertas_set  = set()
    alertas_detalle = {}   # (lat, lon) → probabilidad

    if path_alertas.exists():
        alertas = pd.read_csv(path_alertas)
        alertas['fecha_pronostico'] = pd.to_datetime(alertas['fecha_pronostico']).dt.date
        alertas_hoy = alertas[alertas['fecha_pronostico'] == fecha_verificar]
        for _, row in alertas_hoy.iterrows():
            alertas_set.add((row['lat'], row['lon']))
            alertas_detalle[(row['lat'], row['lon'])] = row.get('probabilidad', None)

    # ── 2. Focos FIRMS reales hoy ──────────────────────────────────
    firms_path = historial_dir / "historial_firms.csv"
    reales_set = set()

    if firms_path.exists():
        firms = pd.read_csv(firms_path)
        firms['fecha'] = pd.to_datetime(firms['fecha']).dt.date
        firms_hoy = firms[(firms['fecha'] == fecha_verificar) & (firms['focos'] > 0)]
        for _, row in firms_hoy.iterrows():
            if row['lat'] is not None and not pd.isna(row['lat']):
                reales_set.add((row['lat'], row['lon']))

    # ── 3. Matriz de confusión del día ────────────────────────────
    aciertos_set = alertas_set & reales_set   # VP
    falsas_set   = alertas_set - reales_set   # FP
    perdidos_set = reales_set  - alertas_set  # FN

    vp = len(aciertos_set)
    fp = len(falsas_set)
    fn = len(perdidos_set)

    precision = vp / (vp + fp) if (vp + fp) > 0 else None
    recall    = vp / (vp + fn) if (vp + fn) > 0 else None

    # ── 4. Guardar en historial_chequeos_diarios.csv ───────────────
    chequeo_path = historial_dir / "historial_chequeos_diarios.csv"
    fila = {
        'fecha':              fecha_str,
        'alertas_emitidas':   len(alertas_set),
        'focos_reales':       len(reales_set),
        'vp':                 vp,
        'fp':                 fp,
        'fn':                 fn,
        'precision':          round(precision, 4) if precision is not None else None,
        'recall':             round(recall,    4) if recall    is not None else None,
    }
    fila_df = pd.DataFrame([fila])

    if chequeo_path.exists():
        hist = pd.read_csv(chequeo_path)
        # Reemplazar si ya existe una fila para esta fecha (re-ejecución)
        hist = hist[hist['fecha'] != fecha_str]
        hist = pd.concat([hist, fila_df], ignore_index=True)
    else:
        hist = fila_df

    hist.to_csv(chequeo_path, index=False)
    print(f"💾 Chequeo guardado: {chequeo_path}")

    # ── 5. Consola ─────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"VERIFICACIÓN — {fecha_str}")
    print(f"{'='*55}")
    print(f"  Alertas emitidas ayer: {len(alertas_set)}")
    print(f"  Focos FIRMS hoy:       {len(reales_set)}")
    print(f"  ✅ Aciertos (VP):      {vp}")
    print(f"  ⚠️  Falsas alarmas (FP): {fp}")
    print(f"  ❌ No detectados (FN): {fn}")
    if precision is not None:
        print(f"  Precisión: {precision:.1%}  |  Recall: {recall:.1%}")
    print(f"{'='*55}")

    if aciertos_set:
        print("  Aciertos:")
        for lat, lon in sorted(aciertos_set):
            depto = depto_map.get((lat, lon), "Desconocido")
            prob  = alertas_detalle.get((lat, lon))
            prob_str = f"  (prob. {prob:.1%})" if prob is not None else ""
            print(f"    • {depto} ({abs(lat):.2f}°S, {abs(lon):.2f}°O){prob_str}")

    if falsas_set:
        print("  Falsas alarmas:")
        for lat, lon in sorted(falsas_set):
            depto = depto_map.get((lat, lon), "Desconocido")
            prob  = alertas_detalle.get((lat, lon))
            prob_str = f"  (prob. {prob:.1%})" if prob is not None else ""
            print(f"    • {depto} ({abs(lat):.2f}°S, {abs(lon):.2f}°O){prob_str}")

    if perdidos_set:
        print("  No detectados:")
        for lat, lon in sorted(perdidos_set):
            depto = depto_map.get((lat, lon), "Desconocido")
            print(f"    • {depto} ({abs(lat):.2f}°S, {abs(lon):.2f}°O)")

    # ── 6. Mensaje Telegram ────────────────────────────────────────
    sin_alertas   = len(alertas_set) == 0
    sin_incendios = len(reales_set)  == 0

    if sin_alertas and sin_incendios:
        mensaje = (
            f"<b>✅ VERIFICACIÓN — {fecha_str}</b>\n\n"
            f"Sin alertas emitidas ayer. Sin focos FIRMS hoy.\n"
            f"<i>Comportamiento correcto para temporada baja.</i>"
        )

    elif sin_alertas and not sin_incendios:
        lineas_perdidos = ""
        for lat, lon in sorted(perdidos_set):
            depto = depto_map.get((lat, lon), "Desconocido")
            lineas_perdidos += f"  • {depto} ({abs(lat):.2f}°S, {abs(lon):.2f}°O)\n"
        mensaje = (
            f"<b>❌ VERIFICACIÓN — {fecha_str}</b>\n\n"
            f"Sin alertas emitidas ayer, pero hubo {len(reales_set)} foco(s) real(es) hoy:\n"
            f"{lineas_perdidos}\n"
            f"<i>El sistema no detectó estos incendios. Revisar umbral o features.</i>"
        )

    elif not sin_alertas and sin_incendios:
        lineas_falsas = ""
        for lat, lon in sorted(falsas_set):
            depto = depto_map.get((lat, lon), "Desconocido")
            prob  = alertas_detalle.get((lat, lon))
            prob_str = f" ({prob:.1%})" if prob is not None else ""
            lineas_falsas += f"  • {depto} ({abs(lat):.2f}°S, {abs(lon):.2f}°O){prob_str}\n"
        mensaje = (
            f"<b>⚠️ VERIFICACIÓN — {fecha_str}</b>\n\n"
            f"Se emitieron {len(alertas_set)} alerta(s) ayer, pero no hubo focos FIRMS hoy.\n\n"
            f"Falsas alarmas:\n{lineas_falsas}\n"
            f"<i>Precisión del día: 0%</i>"
        )

    else:
        # Hubo alertas y hubo incendios — caso más informativo
        lineas_aciertos = ""
        for lat, lon in sorted(aciertos_set):
            depto = depto_map.get((lat, lon), "Desconocido")
            prob  = alertas_detalle.get((lat, lon))
            prob_str = f" ({prob:.1%})" if prob is not None else ""
            lineas_aciertos += f"  • {depto} ({abs(lat):.2f}°S, {abs(lon):.2f}°O){prob_str}\n"

        lineas_falsas = ""
        for lat, lon in sorted(falsas_set):
            depto = depto_map.get((lat, lon), "Desconocido")
            prob  = alertas_detalle.get((lat, lon))
            prob_str = f" ({prob:.1%})" if prob is not None else ""
            lineas_falsas += f"  • {depto} ({abs(lat):.2f}°S, {abs(lon):.2f}°O){prob_str}\n"

        lineas_perdidos = ""
        for lat, lon in sorted(perdidos_set):
            depto = depto_map.get((lat, lon), "Desconocido")
            lineas_perdidos += f"  • {depto} ({abs(lat):.2f}°S, {abs(lon):.2f}°O)\n"

        prec_str = f"{precision:.1%}" if precision is not None else "—"
        rec_str  = f"{recall:.1%}"    if recall    is not None else "—"

        mensaje = f"<b>📋 VERIFICACIÓN — {fecha_str}</b>\n\n"
        mensaje += f"Alertas emitidas ayer: {len(alertas_set)}  |  Focos FIRMS hoy: {len(reales_set)}\n\n"

        if aciertos_set:
            mensaje += f"✅ <b>Aciertos ({vp}):</b>\n{lineas_aciertos}\n"
        if falsas_set:
            mensaje += f"⚠️ <b>Falsas alarmas ({fp}):</b>\n{lineas_falsas}\n"
        if perdidos_set:
            mensaje += f"❌ <b>No detectados ({fn}):</b>\n{lineas_perdidos}\n"

        mensaje += f"<b>Precisión del día: {prec_str}  |  Recall del día: {rec_str}</b>"

    enviar_telegram(mensaje)
    return fila


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # argparse no funciona en Jupyter: el kernel inyecta sus propios argumentos
    # (-f kernel-xxx.json) que argparse rechaza con SystemExit.
    # Se detecta el entorno y se usa una variable manual en ese caso.
    en_jupyter = 'ipykernel' in sys.modules

    if en_jupyter:
        # Ajustar FECHA_MANUAL para verificar un día específico.
        # Dejar en None para verificar hoy.
        FECHA_MANUAL = None   # ej: '2026-06-25'
        fecha = (datetime.strptime(FECHA_MANUAL, '%Y-%m-%d').date()
                 if FECHA_MANUAL else datetime.now().date())
    else:
        import argparse
        parser = argparse.ArgumentParser(
            description="Verifica la predicción del día anterior contra focos FIRMS reales."
        )
        parser.add_argument(
            '--fecha',
            default=None,
            help="Fecha a verificar en formato YYYY-MM-DD (default: hoy)"
        )
        args  = parser.parse_args()
        fecha = (datetime.strptime(args.fecha, '%Y-%m-%d').date()
                 if args.fecha else datetime.now().date())

    chequear(fecha)

