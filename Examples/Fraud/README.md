# Proyecto de Detección de Fraude con Redes Neuronales

Este proyecto implementa un modelo de red neuronal para la detección de fraudes en transacciones de tarjeta de crédito, utilizando el famoso dataset de Kaggle [`creditcard.csv`](https://www.kaggle.com/mlg-ulb/creditcardfraud). Se incluyen herramientas de visualización, métricas y análisis de performance comparado con regresión logística.

---

## 📁 Estructura del Proyecto

```
fraud_detection/
│
├── data/                    # Datos originales y procesados
│
├── notebooks/               # Notebooks principales
│   ├── EDA.ipynb            # Análisis exploratorio de datos
│   ├── RedNeuronal.ipynb    # Entrenamiento y visualización DNN
│   ├── Evaluacion.ipynb     # Comparación con regresión logística
│   └── runColab.ipynb       # Versión compacta para Google Colab
│
├── src/                     # Módulos reutilizables
│   ├── preprocess.py        # Procesamiento de datos, reducción y balanceo
│   ├── train.py             # Entrenamiento de modelos
│   ├── evaluate.py          # Métricas y visualización de performance
│   ├── model.py             # Arquitecturas disponibles
│   └── utils.py             # Funciones auxiliares (plots, configuraciones)
│
├── models/                  # Modelos entrenados en formato .keras
│
├── config.yaml              # Parámetros del modelo, entrenamiento y paths
├── requirements.txt         # Librerías necesarias
└── README.md                # Este archivo
```

---

## ⚙️ Ejecución del proyecto

### 🔧 1. Crear y activar entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 🧪 2. Procesar datos

```bash
python src/preprocess.py
```

Esto genera los archivos procesados balanceados o completos, según la configuración en `config.yaml`.

---

## 📦 Archivos generados por `preprocess.py`

El script `src/preprocess.py` genera diferentes conjuntos de datos dependiendo de la configuración en `config.yaml`.

### 🛠️ Configuración relevante en `config.yaml`

```yaml
preprocessing:
  sample_size: 10000       # Tamaño total del dataset (puede ser None para usar todo)
  subsample: true          # True = crear dataset balanceado para entrenamiento
```

### 📁 Archivos generados en `data/processed/`

| Archivo               | Descripción |
|-----------------------|-------------|
| `train_full.csv`      | Dataset de entrenamiento completo, con clases desbalanceadas (para usar con `class_weight`) |
| `train.csv`           | Dataset balanceado por submuestreo (fraude = no fraude), generado si `subsample=True` |
| `val.csv`             | Conjunto de validación, común para ambos casos |
| `test.csv`            | Conjunto de test sobre los datos originales, desbalanceado |
| `test_balanced.csv`   | Conjunto de test balanceado, generado solo si `subsample=True` |

✅ Esto permite entrenar con dos flujos:
- Dataset balanceado (`train.csv`) sin `class_weight`
- Dataset completo (`train_full.csv`) usando `class_weight=True`

Y evaluar en:
- Test realista (`test.csv`)
- Test balanceado (`test_balanced.csv`)

---

## 🧠 Parámetros de configuración (`config.yaml`)

```yaml
model:
  model_type: deep          # Opciones: simple, deep
  units1: 32
  units2: 16
  dropout: 0.3
  l2: 0.001                 # Regularización L2
  optimizer: adam

training:
  epochs: 50
  batch_size: 128
  use_class_weight: true    # Ponderación por clase
  use_early_stopping: true  # Detención temprana

preprocessing:
  sample_size: 1000000      # Número total de muestras a usar
  subsample: false          # Usar undersampling balanceado
  random_state: 42

evaluation:
  use_balanced_test: false
```

---

## 📊 Visualizaciones disponibles

- Matriz de correlación de variables (`plot_correlation_matrix`)
- Score vs Threshold con F1-score (`plot_f1_vs_threshold`)
- Curva ROC comparativa
- KS test para señal/background (train vs test)
- Distribución de scores por clase (`plot_score_distribution`)
- Matriz de confusión con métricas

---

## 📎 Notas

- El dataset original está desbalanceado (solo 492 fraudes sobre 284,000 transacciones).
- Puedes ajustar el threshold de decisión (`0.5 → 0.95`) para priorizar recall o precisión.
- El modelo de red neuronal puede sobreajustarse si no se usa regularización o `early stopping`.

---

## ▶️ Ejecutar en Google Colab

Puedes ejecutar todo el proyecto directamente desde Google Colab haciendo clic en el siguiente botón:

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/solivare/DeepNeuralNetworkUSS/blob/main/Examples/Fraud/notebooks/runColab.ipynb)

---
