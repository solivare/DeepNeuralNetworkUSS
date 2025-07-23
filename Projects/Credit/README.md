# 🧠 Credit Scoring con Redes Neuronales

Este proyecto utiliza técnicas de aprendizaje profundo para predecir si una persona caerá en mora en los próximos 2 años, usando el dataset **Give Me Some Credit**. Es parte del curso de Deep Neural Networks dictado por **Sebastián Olivares** en la Universidad San Sebastián.

## 📌 Objetivo

Desarrollar un modelo predictivo utilizando una red neuronal que permita clasificar clientes en riesgo de default, comparando su desempeño con una regresión logística tradicional.

## 📊 Dataset

- [Give Me Some Credit (Kaggle)](https://www.kaggle.com/c/GiveMeSomeCredit)
- Contiene variables como ingreso mensual, antigüedad laboral, número de dependientes, entre otras.
- Columnas:
  - `SeriousDlqin2yrs` (target): si el cliente tuvo pagos vencidos en los próximos 2 años.
  - `RevolvingUtilizationOfUnsecuredLines`, `DebtRatio`, `MonthlyIncome`, etc.

## 🗂️ Estructura del Proyecto

```
Projects/Credit/
├── data/                  # Datos originales y procesados
├── models/                # Modelos entrenados (.keras)
├── notebooks/             # Análisis y experimentación
│   ├── EDA.ipynb
│   ├── RedNeuronal.ipynb
│   ├── Evaluacion.ipynb
│   └── runColab.ipynb     # ✅ Notebook ejecutable en Colab
├── src/                   # Código fuente modular
│   ├── preprocess.py
│   ├── train.py
│   ├── model.py
│   ├── evaluate.py
│   └── utils.py
├── config.yaml            # Hiperparámetros
└── requirements.txt       # Dependencias
```

## 🚀 Ejecutar en Google Colab

Haz clic para abrir y ejecutar el proyecto completo en Colab:

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/solivare/DeepNeuralNetworkUSS/blob/main/Projects/Credit/notebooks/runColab.ipynb)

## 🧪 Ejecución local

```bash
# Crear entorno virtual (opcional)
python -m venv venv
source venv/bin/activate  # o venv\Scripts\activate en Windows

# Instalar dependencias
pip install -r requirements.txt

# Preprocesar datos
python src/preprocess.py

# Entrenar red neuronal
# (desde notebooks/RedNeuronal.ipynb)

# Comparar con regresión logística
# (desde notebooks/Evaluacion.ipynb)
```

## 👨‍🏫 Docente

Sebastián Olivares  
sebastian.olivares@uss.cl  
Universidad San Sebastián  
Curso: Deep Neural Networks – Postgrado 2025
