
# Evaluación Comparativa: Red Neuronal vs Regresión Logística

Este módulo contiene el análisis comparativo entre una red neuronal entrenada previamente y un modelo de regresión logística simple, aplicados a un problema de clasificación binaria.

## Contenido

- `train.py`: Script de entrenamiento y carga de datos.
- `evaluate.py`: Funciones auxiliares para evaluar modelos y generar gráficos de comparación.
- `evaluacion.ipynb`: Notebook donde se realiza:
  - Entrenamiento del modelo de regresión logística.
  - Carga y predicción con una red neuronal preentrenada.
  - Comparación de métricas de desempeño (Precision, Recall, F1-score, ROC AUC).
  - Visualización de curvas ROC y Precision-Recall.
  - Búsqueda del umbral óptimo que maximiza el F1-score para cada modelo.
  - Comparación final usando estos umbrales óptimos.

## Resultados Clave

- Se encontró que la **red neuronal supera a la regresión logística** en todas las métricas principales (especialmente en F1-score y ROC AUC) al ajustar el umbral de decisión.
- Umbrales óptimos encontrados:
  - Red Neuronal: 0.22
  - Regresión Logística: 0.12

## Cómo usar

1. Asegúrate de tener los datos y modelos en las rutas esperadas (`data/cs-training.csv`, `models/nn_model.keras`).
2. Corre el notebook `evaluacion.ipynb` para replicar todo el análisis y las visualizaciones.
3. Puedes ajustar los umbrales en la celda correspondiente si deseas explorar otros valores.

---

Desarrollado para fines académicos y comparativos.
