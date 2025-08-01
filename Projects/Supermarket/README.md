# 🛒 Clasificación de Productos de Retail con CNN

Este proyecto utiliza redes neuronales convolucionales (CNN) para clasificar imágenes de frutas, verduras y productos empaquetados, utilizando el dataset **Retail Product Checkout (RPC)**.

### 📁 Dataset
- Fuente original: [RPC Dataset (Kaggle)](https://www.kaggle.com/datasets/shazadudwadia/retail-product-checkout-dataset)
- Para este ejemplo utilizamos una carpeta preprocesada con las imágenes divididas en `train/`, `val/` y `test/`:  
  📎 [Google Drive - Imágenes Preprocesadas](https://drive.google.com/drive/folders/122i7cJlxN_fXGXcpMJYfd4dmJohi7t_V?usp=drive_link)

---

### 📂 Estructura del Proyecto

```
Examples/RetailProductClassification/
├── data/                        # Carpeta de imágenes preprocesadas (link en Drive)
├── models/                      # Modelos entrenados
├── notebooks/                   # Jupyter Notebooks de análisis
│   ├── EDA.ipynb
│   ├── RedNeuronal.ipynb
│   └── Evaluacion.ipynb
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── config.yaml
└── requirements.txt
```

---

### 🚀 Ejecutar en Google Colab

Haz clic para abrir y ejecutar el proyecto en Colab:

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/solivare/DeepNeuralNetworkUSS/blob/main/Examples/RetailProductClassification/notebooks/runColab.ipynb)

---

### 🧪 Ejecución Local

```bash
# Crear entorno virtual (opcional)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Preprocesar datos (opcional, ya se entregan preprocesados)
python src/preprocess.py

# Entrenamiento y evaluación
# Ver notebooks/RedNeuronal.ipynb y Evaluacion.ipynb
```

---

### 👨‍🏫 Docente

Sebastián Olivares  
Curso: Deep Neural Networks – Postgrado 2025  
Universidad San Sebastián  
sebastian.olivares@uss.cl
