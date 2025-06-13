# 🔍 Proyecto de Deep Learning – Predicción de Abandono de Clientes (Churn)

Este proyecto entrena una red neuronal para predecir si un cliente abandonará un servicio, basado en variables como cantidad de compras, contacto con soporte técnico, días inactivo, etc.

---

## 📁 Estructura del Proyecto

```
ChurnNN/
├── notebooks/
│   └── main.ipynb              # Notebook con entrenamiento y evaluación
├── src/
│   ├── model.py                   # Definición del modelo Keras
│   ├── utils.py                   # Funciones de visualización y métricas
│   └── config.yaml                # Configuración editable del modelo
├── data/
│   └── churn_dataset.csv # Dataset simulado
├── setup.sh                       # Script para preparar entorno local
├── requirements.txt               # Librerías necesarias
└── README.md                      # Este archivo
```

---

## 🧪 ¿Cómo correr este proyecto?

### ✅ Opción 1: Google Colab (RECOMENDADA)

## 🚀 Ejecución del Proyecto en Google Colab

Este proyecto puede ejecutarse fácilmente desde **Google Colab**. Solo sigue los pasos en orden:

### 🔧 1. Preparar el entorno

Abre el siguiente notebook para clonar el repositorio, instalar las dependencias y configurar los paths automáticamente:

[![Abrir 0.Setup.ipynb en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/solivare/DeepNeuralNetworkUSS/blob/main/Examples/ChurnNN/notebooks/0.Setup.ipynb)

📁 Archivo: `notebooks/0.Setup.ipynb`

---

### 🧠 2. Ejecutar el modelo de churn

Una vez preparado el entorno, abre el notebook principal donde se realiza todo el flujo de entrenamiento, visualización y evaluación del modelo de churn.

[![Abrir 1.Main.ipynb en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/solivare/DeepNeuralNetworkUSS/blob/main/Examples/ChurnNN/notebooks/1.Main.ipynb)

📁 Archivo: `notebooks/1.Main.ipynb`

---

> ⚠️ **Importante:** siempre debes ejecutar primero `0.Setup.ipynb` antes de cualquier otro notebook del proyecto, para que las rutas y dependencias estén correctamente configuradas.

---

### 💻 Opción 2: Ejecutar en tu computador (VS Code / Jupyter)

1. Clona este repositorio:
   ```bash
   git clone https://github.com/solivare/DeepNeuralNetworkUSS.git
   cd Examples/ChurnNN
   ```

2. Corre el script de configuración:
   ```bash
   bash setup.sh
   ```

3. Activa el entorno virtual:
   - En Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
   - En Windows:
     ```cmd
     venv\Scripts\activate
     ```

4. Abre `notebooks/main.ipynb` y ejecútalo paso a paso.

## 📦 ¿Qué se espera de ti?

- Leer y entender cada bloque del notebook
- Ejecutar el modelo y visualizar resultados
- Subir tu versión modificada a tu repositorio de GitHub
- Documentar tus cambios en el `README.md` personal

---

📬 ¿Dudas? Contacta al profesor o deja un issue en el repositorio.


