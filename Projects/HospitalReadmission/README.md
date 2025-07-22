# Predicción de Readmisión Hospitalaria con Redes Neuronales

Este proyecto forma parte del curso **Deep Neural Networks** impartido en la **Universidad San Sebastián** por el profesor **Sebastián Olivares** (sebastian.olivares@uss.cl). El objetivo es explorar el uso de modelos de Machine Learning (ML) y Deep Learning (DL) para predecir si un paciente será readmitido en un hospital dentro de los próximos 30 días luego del alta médica.

---

## Descripción del Proyecto

Los estudiantes trabajarán con un conjunto de datos clínicos reales que contiene registros de **más de 100 mil pacientes** diabéticos atendidos entre 1999 y 2008 en 130 hospitales de EE.UU. El objetivo es **predecir la readmisión hospitalaria temprana** (`<30 días`), un problema crítico para la gestión eficiente del sistema de salud y la reducción de costos hospitalarios.

Este proyecto busca que los alumnos:

- Comprendan el flujo completo de un pipeline de ML/DL.
- Analicen datos clínicos reales con técnicas de EDA.
- Implementen y comparen modelos clásicos (regresión logística) y redes neuronales profundas.
- Evalúen los modelos utilizando métricas apropiadas.
- Observen el efecto del sobreajuste y cómo mitigarlo.

---

## Estructura del Proyecto

```
HospitalReadmission/
│
├── config.yaml                # Parámetros del modelo y del entrenamiento
├── src/
│   ├── train.py               # Entrenamiento de red neuronal
│   ├── evaluate.py           # Funciones de evaluación y visualización
│   └── preprocess.py         # Limpieza y codificación de datos originales
│
├── data/
│   ├── diabetic_data.csv      # Dataset original de UCI (no subido al repo)
│   └── hospital_readmission_clean.csv  # Dataset limpio generado
│
├── notebooks/
│   ├── EDA.ipynb            # Análisis exploratorio
│   ├── ModeloBase.ipynb     # Modelo de regresión logística
│   ├── RedNeuronal.ipynb    # Entrenamiento con Keras
│   └── Evaluacion.ipynb     # Comparación de modelos y métricas
```

---

## Acerca del Dataset

- **Fuente oficial UCI**:  
  https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

- **Versión en Kaggle**:  
  https://www.kaggle.com/datasets/whenamancodes/diabetes-patient-readmission-prediction

Este dataset incluye variables como:
- Edad, género, raza.
- Diagnósticos (códigos ICD-9).
- Medicamentos administrados.
- Número de visitas previas, ingresos y procedimientos.
- Variable objetivo: `readmitted` (`<30` → 1, otra cosa → 0).

---

## Cómo ejecutar el proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/solivare/DeepNeuralNetworkUSS.git
cd DeepNeuralNetworkUSS/Projects/HospitalReadmission
```

### 2. Crear un entorno virtual y activar

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar el dataset original

Debes descargar manualmente el archivo `diabetic_data.csv` desde el sitio de UCI y colocarlo en la carpeta `data/`.

---

## Flujo sugerido para los estudiantes

1. Ejecutar `preprocess.py` para obtener el dataset limpio.
2. Explorar los datos en `EDA.ipynb`.
3. Entrenar y evaluar una regresión logística en `ModeloBase.ipynb`.
4. Entrenar una red neuronal en `RedNeuronal.ipynb` (con early stopping desactivado para ver sobreajuste).
5. Comparar ambos modelos en `Evaluacion.ipynb`.

---

## Temas pedagógicos abordados

- Preprocesamiento de datos clínicos reales.
- Desbalance de clases y su impacto.
- Métricas apropiadas para clasificación binaria.
- Regularización (`L2`) en redes neuronales.
- Interpretación de curvas ROC y Precision-Recall.
- Modularización del código en proyectos de ML reales.

---

## Ideas adicionales para estudiantes

- Implementar `EarlyStopping` y `Dropout` para reducir el sobreajuste.
- Explorar técnicas de reponderación o submuestreo para desbalance de clases.
- Hacer tuning de hiperparámetros desde el archivo `config.yaml`.
- Agregar interpretabilidad con SHAP o LIME.

---

📬 Para dudas o sugerencias, puedes contactar a:
**Sebastián Olivares** – sebastian.olivares@uss.cl

---

## 🚀 Ejecutar en Google Colab

Puedes ejecutar este proyecto directamente en Google Colab haciendo clic en el siguiente botón:

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/solivare/DeepNeuralNetworkUSS/blob/main/Projects/HospitalReadmission/notebooks/runColab.ipynb)
