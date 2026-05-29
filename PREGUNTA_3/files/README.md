# Detección de Cumplimiento de EPP en Obras de Construcción (YOLOv9)

**Pregunta 3 — Examen Parcial · Redes Neuronales y Aprendizaje Profundo**
Maestría en Inteligencia Artificial (2026-1), Escuela de Posgrado — UNI
Grupo N.º 1 · Docente: Ph.D. Aldo Camargo

Entrenamiento y evaluación de un detector **YOLOv9 (GELAN-E)** para verificar el
uso de casco de seguridad en obras de construcción, con un **verificador de
cumplimiento basado en reglas** y un análisis crítico de robustez y limitaciones.

---

## 1. Estructura del repositorio

```
.
├── yolov9_helmet_detection.ipynb   # Notebook principal (end-to-end)
├── compliance_checker.py           # Verificador de cumplimiento (módulo, stateless)
├── requirements.txt                # Especificación de entorno
├── informe_pregunta3.pdf           # Informe en formato NeurIPS
└── exports/                        # Artefactos generados (métricas, figuras, CSV)
    ├── metrics_global.csv
    ├── metrics_per_class.csv
    ├── ablation_table.csv
    ├── robustness_by_occlusion.csv
    ├── confusion_matrix.png
    ├── training_curves.png
    ├── robustness_report.png
    ├── compliance_demo.png
    └── compliance_report.csv / session_summary.csv
```

## 2. Reproducción de extremo a extremo (comando único)

El pipeline completo está en el notebook y se ejecuta de principio a fin con un
solo comando documentado:

```bash
# 1) Crear entorno
pip install -r requirements.txt

# 2) Ejecutar TODO el pipeline (descarga datos, entrena, evalúa, genera artefactos)
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 yolov9_helmet_detection.ipynb
```

> **Requisitos previos:** GPU CUDA (probado en T4, 15.6 GB) y una clave de
> Roboflow. Definir la variable de entorno antes de ejecutar:
> `export ROBOFLOW_KEY=tu_clave`  (o usar *Colab → Secrets → ROBOFLOW_KEY*).
> En Colab, abrir el notebook y ejecutar *Runtime → Run all*.

El notebook descarga el **Safety Helmet Wearing Dataset (SHWD v3)** desde
Roboflow, reconcilia las clases al esquema unificado
`{0: helmet, 1: no_helmet, 2: person}`, ejecuta la ablación de *mosaic*, entrena
el modelo final, evalúa sobre el conjunto de prueba y escribe todos los
artefactos en `exports/`.

## 3. Decisiones de entrenamiento (resumen)

| Aspecto | Decisión |
|---|---|
| Modelo final | YOLOv9-e (GELAN-E), 100 épocas |
| Resolución de entrada | 640×640 |
| Anclas | *anchor-free* (GELAN + DFL); sin *clustering* de anclas |
| Mosaic | **ON** (ganó la ablación A vs B por +0.0045 mAP@0.5) |
| Desbalance de clase | `cls=3.0` + `label_smoothing=0.1` |
| Reproducibilidad | semilla=42, `patience=20` |

## 4. Resultados (conjunto de prueba)

| Métrica | Valor |
|---|---|
| mAP@0.5 (global) | **0.929** |
| mAP@0.5:0.95 (global) | 0.583 |
| Precisión / Recall (global) | 0.934 / 0.887 |

## 5. ⚠️ Hallazgo crítico (leer antes de cualquier despliegue)

La versión del dataset usada **no contiene anotaciones de la clase
`no_helmet`**: la matriz de confusión muestra que el *ground truth* de prueba
solo contiene `helmet` y `person`. En consecuencia:

1. La fila "no_helmet" de `metrics_per_class.csv` corresponde **en realidad a la
   clase `person`** (desalineación de índices por `ap_class_index`).
2. El verificador, con `flag_person=False`, **no marca** a los trabajadores sin
   casco (rotulados como `person`), por lo que reporta "100 % de cumplimiento"
   por la vía de **falsos negativos**.

**Acción requerida:** reentrenar con un dataset que incluya negativos explícitos
(`no_helmet`) o tratar la clase `person`/cabeza como no conforme mediante una
regla de asociación cabeza–casco; luego recalibrar el umbral priorizando el
*recall* de la clase no conforme. Ver Secciones 5–8 del informe.

## 6. Verificador de cumplimiento

`compliance_checker.py` es un módulo independiente y *stateless*. Reglas
configurables sin reentrenar (`COMPLIANCE_RULES`): umbral de confianza, qué
clases marcan no conformidad, colores de anotación. Expone `process_frame`,
`annotate_frame` y `summarize_session`.

## 7. Datos y licencia

- **SHWD** — njvisionpower / Roboflow Universe (v3). Licencia del export: CC BY 4.0.
- Código de este repositorio: uso académico (curso UNI 2026-1).
