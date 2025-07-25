
# Tarea 2 - Detección de Fraude con Redes Neuronales

Este proyecto implementa un modelo de red neuronal para detectar fraudes en transacciones con tarjetas de crédito, usando el dataset de Kaggle [`creditcard.csv`](https://www.kaggle.com/mlg-ulb/creditcardfraud). Se compara el rendimiento con un modelo de regresión logística.

---

## 📁 Estructura de la Carpeta Entregada

```
Tarea2-JuanBustamante/
│
├── config.yaml              # Parámetros del modelo y configuración general
├── requirements.txt         # Librerías necesarias
├── README.md                # Este archivo
│
├── models/                  # Modelos entrenados en formato .keras
│   └── model.keras
│
├── src/                     # Código fuente del entrenamiento
│   ├── train.py
│   ├── model.py
│   ├── evaluate.py
│   ├── preprocess.py
│   └── utils.py
│
├── notebooks/               # Notebook con resultados
│   └── evaluacion.ipynb
```

---

## ⚙️ Instrucciones de Ejecución

### 1️⃣ Crear entorno virtual y activar (opcional pero recomendado)

```bash
python -m venv venv
venv\Scripts\activate      # En Windows
pip install -r requirements.txt
```

### 2️⃣ Entrenar el modelo

```bash
python src/train.py
```

Esto genera el modelo `model.keras` en la carpeta `models/`.

### 3️⃣ Evaluar el modelo

Abre y ejecuta el notebook:

```
notebooks/evaluacion.ipynb
```

---

## 🧪 Métricas y Visualizaciones Incluidas

- Matriz de confusión
- Curva ROC
- F1-score vs Threshold
- Distribución de predicciones
- Comparación con regresión logística
- Test KS para sobreajuste

---

## 📌 Notas

- El modelo fue ajustado con `early stopping`, `dropout` y posibilidad de usar `class_weight`.
- Se optimizó el threshold de decisión para maximizar el F1-score.
- El dataset original está severamente desbalanceado (menos del 0.2% de fraudes).

---

## 👨‍💻 Autor

Juan Bustamante Castillo  
Universidad San Sebastián  
Tarea 2 - Curso de Redes Neuronales
