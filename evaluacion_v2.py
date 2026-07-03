#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
evaluacion_v2.py
================
Cambios respecto a la versión anterior:

  1. Fix KeyError 'f1': se verifica que depto_stats no esté vacío antes de
     construir el DataFrame y llamar a sort_values. Ocurría cuando no había
     ningún incendio real en el período evaluado (primeros días de operación).

  2. VN a nivel celda-día: la versión anterior contaba un VN por día completo
     si no había ni alerta ni incendio, independientemente de cuántas celdas
     se monitoreaban. Ahora se reconstruye la matriz de confusión completa
     a nivel celda-día usando el registro de celdas monitoreadas por ejecución
     (historial_ejecuciones.csv, columna celdas_monitoreadas) y el grid completo
     (grid_salta_completo.parquet). Esto produce métricas comparables con el
     backtesting, que también opera a nivel celda-día.

  3. Diagnóstico estacional: se agrega una tabla de alertas y aciertos por mes
     en el período evaluado. Útil para detectar si el modelo está generando
     más alertas en los meses correctos (temporada de incendios).

  4. Mensaje Telegram más compacto cuando no hay incendios reales: en lugar de
     omitir la sección de métricas por celda, informa explícitamente cuántos
     VN celda-día se acumularon (señal de que el sistema está corriendo bien
     en temporada baja).

Lógica de comparación (sin cambios):
  - PREDICCIONES: historial_alertas.csv  → qué celdas se alertaron para cada día
  - VERDAD:       historial_firms.csv    → qué celdas tuvieron focos FIRMS ese día
  - EJECUCIONES:  historial_ejecuciones.csv → qué días corrió el sistema
  El sistema predice para mañana y guarda los focos de hoy.
  evaluacion.py compara la predicción del día D con los focos del día D.
"""

import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
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
        print("✅ Mensaje enviado")
    except Exception as e:
        print(f"❌ Error: {e}")


# ──────────────────────────────────────────────────────────────────
def evaluar():

    # ── 1. Días en que corrió el sistema ──────────────────────────
    log_path = ERA5_DIR / "historial" / "historial_ejecuciones.csv"
    if not log_path.exists():
        print("No hay historial de ejecuciones. Correr alerta_incendios.py al menos una vez.")
        return

    ejecuciones = pd.read_csv(log_path, on_bad_lines='warn')
    ejecuciones['fecha_pronostico'] = pd.to_datetime(ejecuciones['fecha_pronostico']).dt.date
    ejecuciones = ejecuciones.drop_duplicates(subset='fecha_pronostico', keep='last')
    fechas_sistema = set(ejecuciones['fecha_pronostico'])
    print(f"Días con ejecución registrada: {len(fechas_sistema)}")
    print(f"  ({min(fechas_sistema)} a {max(fechas_sistema)})")

    # Mapa fecha → n_celdas_monitoreadas (para calcular VN celda-día)
    celdas_por_fecha = ejecuciones.set_index('fecha_pronostico')['celdas_monitoreadas'].to_dict()

    # ── 2. Predicciones emitidas ───────────────────────────────────
    path_alertas     = ERA5_DIR / "historial" / "historial_alertas.csv"
    alertas_por_fecha = {}
    alertas           = pd.DataFrame()

    if path_alertas.exists():
        alertas = pd.read_csv(path_alertas)
        if not alertas.empty:
            alertas['fecha_pronostico'] = pd.to_datetime(alertas['fecha_pronostico']).dt.date
            for fecha, grupo in alertas.groupby('fecha_pronostico'):
                alertas_por_fecha[fecha] = set(zip(grupo['lat'], grupo['lon']))

    n_dias_con_alerta = sum(1 for f in fechas_sistema if f in alertas_por_fecha)
    print(f"Días con al menos una alerta emitida: {n_dias_con_alerta}")

    # ── 3. Verdad: focos FIRMS reales ─────────────────────────────
    firms_path = ERA5_DIR / "historial" / "historial_firms.csv"
    if not firms_path.exists():
        print("⚠️ No hay historial_firms.csv todavía. "
              "Se necesita al menos una corrida de alerta_incendios.py.")
        return

    firms = pd.read_csv(firms_path)
    firms['fecha'] = pd.to_datetime(firms['fecha']).dt.date
    firms_reales   = firms[firms['focos'] > 0].copy()
    incendios_por_fecha = {}
    for fecha, grupo in firms_reales.groupby('fecha'):
        incendios_por_fecha[fecha] = set(zip(grupo['lat'], grupo['lon']))

    # ── 4. Mapa de departamentos ───────────────────────────────────
    grid_path = ERA5_DIR / "grid_salta_completo.parquet"
    if grid_path.exists():
        grid      = pd.read_parquet(grid_path)
        depto_map = grid.set_index(['lat_era5', 'lon_era5'])['departamento'].to_dict()
        # Conjunto completo de celdas del grid (para calcular VN celda-día)
        todas_las_celdas = set(zip(grid['lat_era5'], grid['lon_era5']))
    else:
        top_celdas = pd.read_parquet(ERA5_DIR / "top_celdas.parquet")
        depto_map  = top_celdas.set_index(['lat_era5', 'lon_era5'])['departamento'].to_dict()
        todas_las_celdas = set(zip(top_celdas['lat_era5'], top_celdas['lon_era5']))
        print("⚠️ Usando top_celdas.parquet para departamentos.")

    # ── 5. Construir VP / FP / FN / VN ────────────────────────────
    # Nivel DÍA (igual que antes, para exactitud y especificidad diarias)
    VP_dias = FP_dias = FN_dias = VN_dias = 0

    # Nivel CELDA-DÍA (nuevo en v2)
    # CAMBIO v2: en lugar de contar 1 VN por día sin alertas, se reconstruye
    # cuántas celdas-día fueron verdaderos negativos:
    #   VN_celdas = celdas_monitoreadas_ese_día
    #               - alertas_emitidas_ese_día
    #               - incendios_reales_ese_día_sin_alerta
    # Esto hace las métricas comparables con el backtesting.
    VP_celdas = FP_celdas = FN_celdas = VN_celdas = 0

    resultados      = []
    resultados_mes  = defaultdict(lambda: {'alertas': 0, 'aciertos': 0, 'incendios': 0})

    aciertos_por_depto   = defaultdict(int)
    alertas_falsas_depto = defaultdict(int)
    incendios_por_depto  = defaultdict(int)
    alertas_totales_depto = defaultdict(int)

    aciertos_detalle = []
    falsos_detalle   = []
    perdidos_detalle = []

    for fecha_date in sorted(fechas_sistema):
        alertas_set  = alertas_por_fecha.get(fecha_date, set())
        reales_set   = incendios_por_fecha.get(fecha_date, set())
        n_monitoreadas = int(celdas_por_fecha.get(fecha_date, len(todas_las_celdas)))

        tiene_alerta   = len(alertas_set) > 0
        tiene_incendio = len(reales_set)  > 0

        # Nivel día
        if   tiene_alerta and tiene_incendio:     VP_dias += 1
        elif tiene_alerta and not tiene_incendio: FP_dias += 1
        elif not tiene_alerta and tiene_incendio: FN_dias += 1
        else:                                     VN_dias += 1

        # Nivel celda-día
        aciertos_celdas = alertas_set & reales_set
        falsas_celdas   = alertas_set - reales_set
        perdidos_celdas = reales_set  - alertas_set

        vp = len(aciertos_celdas)
        fp = len(falsas_celdas)
        fn = len(perdidos_celdas)
        # VN = celdas monitoreadas que no fueron alertadas ni tuvieron fuego real
        vn = max(0, n_monitoreadas - vp - fp - fn)

        VP_celdas += vp
        FP_celdas += fp
        FN_celdas += fn
        VN_celdas += vn

        # Acumulados por mes
        mes = fecha_date.month
        resultados_mes[mes]['alertas']   += len(alertas_set)
        resultados_mes[mes]['aciertos']  += vp
        resultados_mes[mes]['incendios'] += len(reales_set)

        # Detalles celda-día
        if not alertas.empty and tiene_alerta:
            alertas_dia = alertas[alertas['fecha_pronostico'] == fecha_date]
            for lat, lon in aciertos_celdas:
                rows = alertas_dia[(alertas_dia['lat'] == lat) & (alertas_dia['lon'] == lon)]
                if not rows.empty:
                    aciertos_detalle.append(rows.iloc[0].to_dict())
            for lat, lon in falsas_celdas:
                rows = alertas_dia[(alertas_dia['lat'] == lat) & (alertas_dia['lon'] == lon)]
                if not rows.empty:
                    falsos_detalle.append(rows.iloc[0].to_dict())
        for lat, lon in perdidos_celdas:
            perdidos_detalle.append({
                'fecha': fecha_date, 'lat': lat, 'lon': lon,
                'departamento': depto_map.get((lat, lon), "Desconocido")
            })

        # Por departamento
        for lat, lon in aciertos_celdas:
            depto = depto_map.get((lat, lon), "Desconocido")
            aciertos_por_depto[depto]    += 1
            alertas_totales_depto[depto] += 1
        for lat, lon in falsas_celdas:
            depto = depto_map.get((lat, lon), "Desconocido")
            alertas_falsas_depto[depto]  += 1
            alertas_totales_depto[depto] += 1
        for lat, lon in perdidos_celdas:
            depto = depto_map.get((lat, lon), "Desconocido")
            incendios_por_depto[depto]   += 1

        precision_dia = vp / len(alertas_set) if len(alertas_set) > 0 else 0
        recall_dia    = vp / len(reales_set)  if len(reales_set)  > 0 else 0
        if tiene_alerta or tiene_incendio:
            resultados.append({
                'fecha':           fecha_date,
                'alertas':         len(alertas_set),
                'incendios_reales': len(reales_set),
                'aciertos':        vp,
                'falsas':          fp,
                'perdidos':        fn,
                'vn_celdas':       vn,
                'precision':       precision_dia,
                'recall':          recall_dia,
            })

    # ── 6. Guardar CSVs de detalle ─────────────────────────────────
    if aciertos_detalle:
        pd.DataFrame(aciertos_detalle).to_csv(ERA5_DIR / "aciertos_detalle.csv", index=False)
        print(f"✅ Aciertos guardados: {len(aciertos_detalle)} registros")
    if falsos_detalle:
        pd.DataFrame(falsos_detalle).to_csv(ERA5_DIR / "falsas_alarmas_detalle.csv", index=False)
        print(f"✅ Falsas alarmas guardadas: {len(falsos_detalle)} registros")
    if perdidos_detalle:
        pd.DataFrame(perdidos_detalle).to_csv(ERA5_DIR / "incendios_no_detectados.csv", index=False)
        print(f"✅ Incendios no detectados guardados: {len(perdidos_detalle)} registros")

    # ── 7. Métricas globales celda-día ────────────────────────────
    total_celdas_dia   = VP_celdas + FP_celdas + FN_celdas + VN_celdas
    incendios_totales  = VP_celdas + FN_celdas
    total_alertas      = VP_celdas + FP_celdas

    precision_global = VP_celdas / total_alertas    if total_alertas   > 0 else 0
    recall_global    = VP_celdas / incendios_totales if incendios_totales > 0 else 0
    f1_global        = (2 * precision_global * recall_global
                        / (precision_global + recall_global)
                        if (precision_global + recall_global) > 0 else 0)
    exactitud_celdas = (VP_celdas + VN_celdas) / total_celdas_dia if total_celdas_dia > 0 else 0
    especif_celdas   = VN_celdas / (VN_celdas + FP_celdas)        if (VN_celdas + FP_celdas) > 0 else 0

    # ── 8. Métricas por departamento ──────────────────────────────
    # CAMBIO v2: se verifica que depto_stats no esté vacío antes de sort_values.
    # En la versión anterior, sort_values('f1') sobre un DataFrame vacío sin esa
    # columna lanzaba KeyError cuando no había ningún incendio en el período.
    deptos      = set(list(aciertos_por_depto)
                      + list(incendios_por_depto)
                      + list(alertas_totales_depto))
    depto_stats = []
    for depto in deptos:
        aciertos_d = aciertos_por_depto.get(depto, 0)
        alertas_d  = alertas_totales_depto.get(depto, 0)
        incendios_d = incendios_por_depto.get(depto, 0)
        prec_d = aciertos_d / alertas_d  if alertas_d  > 0 else 0
        rec_d  = aciertos_d / incendios_d if incendios_d > 0 else 0
        f1_d   = (2 * prec_d * rec_d / (prec_d + rec_d)
                  if (prec_d + rec_d) > 0 else 0)
        depto_stats.append({
            'departamento': depto,
            'alertas':      alertas_d,
            'incendios':    incendios_d,
            'aciertos':     aciertos_d,
            'precision':    prec_d,
            'recall':       rec_d,
            'f1':           f1_d,
        })

    # CAMBIO v2: guard explícito antes de sort_values
    if depto_stats:
        df_depto = (pd.DataFrame(depto_stats)
                    .sort_values('f1', ascending=False)
                    .reset_index(drop=True))
    else:
        df_depto = pd.DataFrame(
            columns=['departamento', 'alertas', 'incendios',
                     'aciertos', 'precision', 'recall', 'f1']
        )

    # ── 9. Métricas a nivel día ────────────────────────────────────
    total_dias     = VP_dias + FP_dias + FN_dias + VN_dias
    exactitud_dias = (VP_dias + VN_dias) / total_dias if total_dias > 0 else 0
    especif_dias   = VN_dias / (VN_dias + FP_dias)    if (VN_dias + FP_dias) > 0 else 0
    recall_dias    = VP_dias / (VP_dias + FN_dias)     if (VP_dias + FN_dias) > 0 else 0
    precision_dias = VP_dias / (VP_dias + FP_dias)     if (VP_dias + FP_dias) > 0 else 0
    f1_dias        = (2 * precision_dias * recall_dias
                      / (precision_dias + recall_dias)
                      if (precision_dias + recall_dias) > 0 else 0)

    primera_fecha = min(fechas_sistema)
    fecha_fin     = max(fechas_sistema)

    # ── 10. Diagnóstico estacional ────────────────────────────────
    # CAMBIO v2: tabla de alertas, aciertos e incendios reales por mes.
    # Permite verificar si el modelo está siendo activo en la temporada correcta
    # (septiembre-noviembre en Salta) y silencioso en temporada baja (junio-agosto).
    nombres_mes = {
        1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
    }
    df_mes = pd.DataFrame([
        {
            'mes':       nombres_mes.get(m, str(m)),
            'alertas':   v['alertas'],
            'aciertos':  v['aciertos'],
            'incendios': v['incendios'],
            'precision': round(v['aciertos'] / v['alertas'],   3) if v['alertas']   > 0 else 0,
            'recall':    round(v['aciertos'] / v['incendios'], 3) if v['incendios'] > 0 else 0,
        }
        for m, v in sorted(resultados_mes.items())
    ])

    # ── 11. Imprimir resumen en consola ───────────────────────────
    print(f"\n{'='*60}")
    print(f"PERÍODO: {primera_fecha} a {fecha_fin} ({total_dias} días)")
    print(f"{'='*60}")

    print(f"\nMÉTRICAS POR DÍA:")
    print(f"  ✅ VP - Alerta emitida + incendio real:  {VP_dias}")
    print(f"  ⚠️  FP - Alerta emitida + sin incendio:  {FP_dias}")
    print(f"  ❌ FN - Sin alerta + incendio real:      {FN_dias}")
    print(f"  🟢 VN - Sin alerta + sin incendio:       {VN_dias}")
    print(f"  Exactitud:         {exactitud_dias:.3f}")
    print(f"  Especificidad:     {especif_dias:.3f}")
    print(f"  Recall diario:     {recall_dias:.3f}")
    print(f"  Precisión diaria:  {precision_dias:.3f}")
    print(f"  F1 diario:         {f1_dias:.3f}")

    print(f"\nMÉTRICAS POR CELDA-DÍA (comparables con backtesting):")
    print(f"  Total celda-días evaluados: {total_celdas_dia:,}")
    print(f"  VP: {VP_celdas:,}  |  FP: {FP_celdas:,}  |  FN: {FN_celdas:,}  |  VN: {VN_celdas:,}")
    if incendios_totales > 0:
        print(f"  Precisión:     {precision_global:.3f}")
        print(f"  Recall:        {recall_global:.3f}")
        print(f"  F1:            {f1_global:.3f}")
        print(f"  Exactitud:     {exactitud_celdas:.3f}")
        print(f"  Especificidad: {especif_celdas:.3f}")
    else:
        print(f"  Sin incendios reales en el período → {VN_celdas:,} VN celda-día acumulados.")
        print(f"  (Señal correcta para temporada baja: el sistema no genera alertas falsas.)")

    if not df_depto.empty:
        print(f"\nTOP 10 DEPARTAMENTOS POR F1:")
        print(df_depto.head(10).to_string(index=False))

    if not df_mes.empty:
        print(f"\nDISTRIBUCIÓN MENSUAL (diagnóstico estacional):")
        print(df_mes.to_string(index=False))

    print(f"{'='*60}")

    # ── 12. Mensaje Telegram ──────────────────────────────────────
    mensaje = (
        f"<b>📊 EVALUACIÓN DEL SISTEMA DE ALERTA</b>\n"
        f"<i>Predicciones vs. focos FIRMS reales</i>\n\n"
        f"<b>Período:</b> {primera_fecha} a {fecha_fin} ({total_dias} días)\n\n"
        f"<b>📅 Por día:</b>\n"
        f"  ✅ Alerta + incendio real (VP): {VP_dias}\n"
        f"  ⚠️ Alerta + sin incendio (FP):  {FP_dias}\n"
        f"  ❌ Sin alerta + incendio (FN):  {FN_dias}\n"
        f"  🟢 Sin alerta + sin incendio (VN): {VN_dias}\n"
        f"  Exactitud: {exactitud_dias:.1%} | Especificidad: {especif_dias:.1%}\n\n"
    )

    if incendios_totales > 0:
        mensaje += (
            f"<b>🔢 Por celda-día:</b>\n"
            f"  Alertas emitidas:      {total_alertas:,}\n"
            f"  Incendios reales:      {incendios_totales:,}\n"
            f"  VP: {VP_celdas} | FP: {FP_celdas} | FN: {FN_celdas} | VN: {VN_celdas:,}\n"
            f"  Precisión: {precision_global:.1%} | Recall: {recall_global:.1%} | F1: {f1_global:.3f}\n"
            f"  Exactitud: {exactitud_celdas:.1%} | Especificidad: {especif_celdas:.1%}\n\n"
        )
    else:
        # CAMBIO v2: en temporada baja sin incendios, informa los VN celda-día
        # acumulados como señal positiva de que el sistema no genera falsas alarmas.
        mensaje += (
            f"<b>🔢 Por celda-día:</b>\n"
            f"  Sin incendios reales (FIRMS) en el período.\n"
            f"  {VN_celdas:,} celda-días correctamente silenciosos (VN).\n"
            f"  (Temporada baja: comportamiento esperado del modelo.)\n\n"
        )

    if not df_depto.empty and incendios_totales > 0:
        mensaje += "<b>Top 5 departamentos por F1:</b>\n"
        for i, (_, row) in enumerate(df_depto.head(5).iterrows(), start=1):
            mensaje += (
                f"  {i}. {row['departamento']}: "
                f"P {row['precision']:.1%} | R {row['recall']:.1%} | F1 {row['f1']:.3f}\n"
            )
        mensaje += "\n"

    if not df_mes.empty:
        mensaje += "<b>📆 Por mes:</b>\n"
        for _, row in df_mes.iterrows():
            if row['alertas'] > 0 or row['incendios'] > 0:
                mensaje += (
                    f"  {row['mes']}: alertas={int(row['alertas'])} "
                    f"incendios={int(row['incendios'])} "
                    f"aciertos={int(row['aciertos'])}\n"
                )
        mensaje += "\n"

    if perdidos_detalle:
        mensaje += f"<b>⚠️ Incendios no detectados (Top 3):</b>\n"
        for d in perdidos_detalle[:3]:
            mensaje += f"  {d['fecha']}: {d['departamento']} ({abs(d['lat']):.2f}°S, {abs(d['lon']):.2f}°O)\n"
        mensaje += "\n"

    if aciertos_detalle:
        mensaje += f"<b>✅ Aciertos recientes (Top 3):</b>\n"
        for d in aciertos_detalle[:3]:
            depto = d.get('departamento', depto_map.get((d.get('lat'), d.get('lon')), "Desconocido"))
            mensaje += f"  {d.get('fecha_pronostico', '?')}: {depto}\n"
        mensaje += "\n"

    mensaje += f"<i>Fuente: historial_firms.csv (VIIRS NOAA-20)</i>"
    enviar_telegram(mensaje)

    # ── 13. Guardar CSVs de resumen ───────────────────────────────
    df_res = pd.DataFrame(resultados)
    if not df_res.empty:
        df_res.to_csv(ERA5_DIR / "evaluacion_rendimiento.csv", index=False)
    if not df_depto.empty:
        df_depto.to_csv(ERA5_DIR / "evaluacion_por_departamento.csv", index=False)
    if not df_mes.empty:
        df_mes.to_csv(ERA5_DIR / "evaluacion_por_mes.csv", index=False)
    print("✅ Informes guardados en CSV.")


if __name__ == "__main__":
    evaluar()

