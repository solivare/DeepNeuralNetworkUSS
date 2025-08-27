# 🧠 Clasificación de Cáncer de Mama con Deep Learning

Este proyecto utiliza redes neuronales convolucionales (CNN) para detectar cáncer de mama en imágenes histopatológicas, específicamente **carcinoma ductal invasivo (IDC)**. Forma parte del curso de Deep Neural Networks dictado por **Sebastián Olivares** en la Universidad San Sebastián.

## 📌 Objetivo

Construir un modelo de red neuronal capaz de clasificar regiones de tejido mamario como benignas o malignas (IDC), a partir de imágenes de 50x50 píxeles, y visualizar su razonamiento utilizando técnicas como **Grad-CAM**.

## 🧬 Dataset

- 📦 [Breast Histopathology Images (Kaggle)](https://www.kaggle.com/paultimothymooney/breast-histopathology-images)
- Contiene **277,524** imágenes de 50x50 píxeles:
  - `198,738` imágenes sin IDC (clase 0)
  - `78,786` imágenes con IDC (clase 1)
- Cada imagen proviene de una muestra de biopsia escaneada a 40x.
- Formato del nombre: `patient_xX_yY_classC.png`, donde `C = 0` (benigno), `C = 1` (maligno).

**Fuente original**:  
[http://gleason.case.edu/webdata/jpi-dl-tutorial/IDC_regular_ps50_idx5.zip](http://gleason.case.edu/webdata/jpi-dl-tutorial/IDC_regular_ps50_idx5.zip)

**Referencia académica**:  
- A. Cruz-Roa et al., "Accurate and reproducible invasive breast cancer detection in whole-slide images: A Deep Learning approach", SPIE 2014.

## 🗂️ Estructura del Proyecto

```
Examples/BreastCancer/
├── data/                  # Datos originales y organizados en carpetas
├── models/                # Modelos entrenados (.keras)
├── notebooks/             # Análisis y visualización
│   ├── EDA.ipynb
│   ├── RedNeuronal.ipynb
│   ├── Evaluacion.ipynb
│   └── runColab.ipynb     # ✅ Ejecutable en Colab
├── src/                   # Código fuente del modelo
│   ├── preprocess.py
│   ├── train.py
│   ├── model.py
│   ├── evaluate.py
│   └── utils.py
├── config.yaml            # Parámetros del modelo y rutas
└── requirements.txt       # Dependencias
```

## 🚀 Ejecutar en Google Colab

Haz clic en el botón para abrir y ejecutar el proyecto completo en Colab:

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/solivare/DeepNeuralNetworkUSS/blob/main/Examples/BreastCancer/notebooks/runColab.ipynb)

## 🧪 Ejecución local

```bash
# Crear entorno virtual (opcional)
python -m venv venv
source venv/bin/activate  # o venv\Scripts\activate en Windows

# Instalar dependencias
pip install -r requirements.txt

# Preprocesar datos (si no existen)
python src/preprocess.py

# Entrenamiento y evaluación desde notebooks/
```

## 🧠 Características del modelo

- Red convolucional (CNN) con 3 capas Conv2D + MaxPooling.
- Preprocesamiento y **data augmentation** (rotación, traslación, flip horizontal).
- Visualización de errores y regiones activadas con **Grad-CAM**.
- Métricas: accuracy, F1-score, AUC, matriz de confusión, distribución de probabilidades.

## 🧑‍🏫 Docente

Sebastián Olivares  
sebastian.olivares@uss.cl  
Universidad San Sebastián  
Curso: Deep Neural Networks – Postgrado 2025
