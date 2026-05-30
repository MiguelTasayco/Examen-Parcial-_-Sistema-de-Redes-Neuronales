# 🔍 Tire Damage Classification via Texture Recognition

> Clasificación automática de llantas dañadas mediante CNN — Proyecto de Maestría en Inteligencia Artificial

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![timm](https://img.shields.io/badge/timm-1.0.3-blue?style=flat-square)](https://github.com/huggingface/pytorch-image-models)
[![Colab](https://img.shields.io/badge/Google%20Colab-A100%20GPU-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Descripción General

Este proyecto implementa y compara múltiples arquitecturas de redes neuronales convolucionales (CNN) para la **clasificación binaria de llantas vehiculares** a partir del análisis de imágenes de su textura superficial (`normal` vs. `cracked`). El trabajo forma parte de un curso de maestría en IA y cubre el pipeline completo: desde el análisis exploratorio hasta la interpretabilidad del modelo.

**Problema:** La inspección manual de llantas es costosa, subjetiva y no escalable. Un sistema automatizado reduce el riesgo de accidentes al detectar daños de forma rápida y objetiva.

---

## Arquitecturas Implementadas

| Modelo | Tipo | Parámetros | Estrategia |
|---|---|---|---|
| **TireNet** | CNN desde cero | ~2–4M | 5 bloques Conv-BN-ReLU + residuales |
| **EfficientNet-B3** | Transfer Learning | ~12M | Full fine-tuning + LR diferenciado |
| **ResNet-50** | Transfer Learning | ~25.6M | Full fine-tuning + LR diferenciado |

---

## Estructura del Repositorio

```
tire-damage-classification/
│
├── 📓 NB1_CNN_Scratch_TireClassification.ipynb   ← CNN desde cero (TireNet)
├── 📓 NB2_TransferLearning_TireClassification.ipynb  ← EfficientNet-B3 + ResNet-50
│
├── 📄 tire_classification_ieee.tex               ← Paper en formato IEEE (LaTeX)
│
└── 📁 outputs/                                   ← Generado al ejecutar los notebooks
    ├── nb1_cnn_scratch/
    │   ├── best_model.pth
    │   ├── ablation_results.csv
    │   ├── training_history.csv
    │   ├── test_predictions.csv
    │   ├── metrics_summary.csv
    │   ├── confusion_matrix.png
    │   ├── roc_curve.png
    │   ├── imbalance_comparison.csv
    │   ├── gradcam_results/           ← 8 imágenes Grad-CAM
    │   └── failure_analysis/          ← Grid + CSV de fallos
    │
    └── nb2_transfer_learning/
        ├── checkpoints/               ← best_effnet.pth, best_resnet.pth
        ├── ablation_results.csv
        ├── training_history_{effnet,resnet}.csv
        ├── test_predictions_{effnet,resnet}.csv
        ├── final_comparison.csv
        ├── model_comparison_chart.png
        ├── confusion_matrices_tl.png
        ├── roc_curves_tl.png
        ├── gradcam_results/           ← 16 imágenes Grad-CAM
        └── failure_analysis/          ← Grids + CSVs por modelo
```

---

## Dataset

**[Tire Texture Image Recognition](https://www.kaggle.com/datasets/jehanbhathena/tire-texture-image-recognition)** — Kaggle (jehanbhathena)

| Split | Normal | Cracked | Total |
|---|---|---|---|
| Train (80%) | — | — | ~822 |
| Val (20%) | — | — | ~206 |
| Test | — | — | — |

> Los valores exactos se completan tras ejecutar la Sección 1 del NB1 (`eda_summary.csv`).

**Descarga automática** en el notebook mediante `kagglehub`:
```python
import kagglehub
path = kagglehub.dataset_download("jehanbhathena/tire-texture-image-recognition")
```

---

## Requisitos

### Entorno recomendado
- **Google Colab** con GPU A100 o V100 (Colab Pro/Pro+)
- Python 3.10+

### Dependencias (instaladas automáticamente en los notebooks)
```
torch>=2.0
torchvision
timm==1.0.3
grad-cam==1.5.4
kagglehub
scikit-learn
torchmetrics
matplotlib
seaborn
pandas
numpy
Pillow
```

---

## Ejecución

### 1. Orden recomendado

```
NB1_CNN_Scratch  →  NB2_TransferLearning
```

El NB2 importa `metrics_summary.csv` del NB1 desde Google Drive para construir la tabla comparativa de los tres modelos. Ejecuta el NB1 completo primero.

### 2. Pasos en cada notebook

1. Ejecutar **Sección 0** — setup e instalación de dependencias
2. Ejecutar **Sección 1** — descarga del dataset (requiere sesión activa en Kaggle)
3. Ejecutar las secciones restantes en orden
4. La **Sección final** exporta todos los artefactos a `/MyDrive/tire_classification/`

### 3. Primera ejecución en Colab

Al llegar a la celda de descarga del dataset, `kagglehub` solicitará autenticación si no detecta una sesión de Kaggle activa. Ingresar `username` y `API token` desde [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token.

---

## Metodología

### Pipeline general

```
Dataset → EDA → Preprocesamiento → Entrenamiento → Evaluación → Interpretabilidad
                      ↓
           Augmentation (baseline / agresivo)
           WeightedRandomSampler (mitigación desbalance)
                      ↓
              Ablación (lr × loss / backbone × freeze)
                      ↓
              Modelo final (mejor config)
                      ↓
           Métricas + Grad-CAM + Análisis de fallos
```

### Estudio de ablación

**NB1 — TireNet** (4 configuraciones × 15 épocas):

| Factor | Niveles |
|---|---|
| Learning rate | `1e-3`, `1e-4` |
| Función de pérdida | BCE ponderada, Focal Loss (α=0.25, γ=2.0) |

**NB2 — Transfer Learning** (4 configuraciones × 12 épocas):

| Factor | Niveles |
|---|---|
| Backbone | EfficientNet-B3, ResNet-50 |
| Freeze mode | `frozen`, `partial` |

### Mitigación del desbalance de clases

- `WeightedRandomSampler` — sobremuestreo implícito de la clase minoritaria en cada mini-batch
- `pos_weight` en BCE — penalización adicional sobre errores en la clase positiva
- Focal Loss — downweighting automático de ejemplos fáciles

### Interpretabilidad

Grad-CAM sobre la última capa convolucional de cada arquitectura:

| Modelo | Target layer |
|---|---|
| TireNet | `features[-1].conv2` |
| EfficientNet-B3 | `backbone.conv_head` |
| ResNet-50 | `backbone.layer4[-1].conv3` |

> Para modelos con salida escalar (1 logit), usar siempre `ClassifierOutputTarget(0)`.

---

## Métricas de Evaluación

Dado el desbalance de clases, la métrica principal de comparación es el **F1-score macro**:

| Métrica | Descripción |
|---|---|
| Accuracy | Proporción de predicciones correctas |
| Precision (macro) | Media de precisión por clase |
| Recall (macro) | Media de sensibilidad por clase — crítico para detectar daño |
| F1 (macro) | Media armónica de precisión y recall por clase ★ |
| F1 (weighted) | F1 ponderado por soporte de clase |
| AUC-ROC | Área bajo la curva ROC |

---

## Decisiones de Diseño Relevantes

**`ClassifierOutputTarget(0)` en Grad-CAM**
Los tres modelos producen un único logit escalar (salida `(B, 1)`). La librería `pytorch-grad-cam` espera un índice de clase válido dentro del tensor de salida. Para salidas escalares, el único índice válido es siempre `0`, independientemente de la clase predicha.

**Split estratificado con semilla fija**
Ambos notebooks usan `random_state=42` y `test_size=0.20` en `train_test_split`, garantizando que los tres modelos se evalúen sobre exactamente el mismo conjunto de validación y test — condición necesaria para que la comparación sea válida.

**Learning rate diferenciado en full fine-tuning**
El backbone recibe `lr / 10` y la cabeza clasificadora recibe `lr`. Esto preserva las representaciones preentrenadas en ImageNet mientras permite adaptación al dominio de texturas de llantas.

**`verbose` eliminado de `ReduceLROnPlateau`**
El parámetro fue deprecado en PyTorch ≥ 2.4. Los notebooks no lo incluyen.

---

## Referencias Clave

- He et al. (2016) — *Deep Residual Learning for Image Recognition* — ResNet
- Tan & Le (2019) — *EfficientNet: Rethinking Model Scaling for CNNs*
- Lin et al. (2017) — *Focal Loss for Dense Object Detection*
- Selvaraju et al. (2017) — *Grad-CAM: Visual Explanations from Deep Networks*
- Cha et al. (2017) — *Deep Learning-Based Crack Damage Detection using CNNs*

---

## Autores

Proyecto desarrollado como entregable del curso de Maestría en Inteligencia Artificial.

GRUPO 01:
- Paul Bazan H.
- Valeria P. Bulege N.
- Oscar Dante Chaíña A.
- Elias O. Diaz C.
- Miguel A. Tasayco M.

---
