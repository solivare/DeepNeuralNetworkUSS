# 🔍 Proyecto de Deep Learning – Predicción de Abandono de Clientes (Churn)

Este proyecto entrena una red neuronal para predecir si un cliente abandonará un servicio, basado en variables como cantidad de compras, contacto con soporte técnico, días inactivo, etc.

---

## 📁 Estructura del Proyecto

```
churn_example/
├── notebooks/
│   └── 01_main.ipynb              # Notebook con entrenamiento y evaluación
├── src/
│   ├── model.py                   # Definición del modelo Keras
│   ├── utils.py                   # Funciones de visualización y métricas
│   └── config.yaml                # Configuración editable del modelo
├── data/
│   └── churn_dataset_completo.csv # Dataset simulado
├── setup.sh                       # Script para preparar entorno local
├── requirements.txt               # Librerías necesarias
└── README.md                      # Este archivo
```

---

## 🧪 ¿Cómo correr este proyecto?

### ✅ Opción 1: Google Colab (RECOMENDADA)

1. Ve a [https://colab.research.google.com](https://colab.research.google.com)
2. Haz clic en la pestaña “GitHub” y busca:
   ```
   solivare/churn_example
   ```
3. Abre el archivo `notebooks/01_main.ipynb`
4. Ejecuta las celdas que:
   - Clonan el repositorio (si no estás ya en él)
   - Instalan dependencias (`!pip install -r requirements.txt`)
   - Agregan el path para importar desde `src/`

⚠️ En Colab **NO necesitas ejecutar `setup.sh`**. Todo se hace desde celdas.

---

### 💻 Opción 2: Ejecutar en tu computador (VS Code / Jupyter)

1. Clona este repositorio:
   ```bash
   git clone https://github.com/solivare/churn_example.git
   cd churn_example
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

---

## 🧠 ¿Cómo crear tu cuenta de GitHub?

1. Ve a [https://github.com/](https://github.com/)
2. Haz clic en **Sign up**
3. Ingresa tu email y una contraseña segura
4. Elige un nombre de usuario (puedes usar tu nombre real o algo profesional)
5. Verifica tu cuenta por correo
6. Una vez dentro, haz clic en **New repository**
   - Nómbralo por ejemplo `churn_clasificador`
   - Selecciona "Public" o "Private"
   - No inicialices con README si vas a subir tu código desde Colab

Ahora puedes subir tus archivos manualmente o usar comandos de Git desde Colab o tu computador.

---

## 📦 ¿Qué se espera de ti?

- Leer y entender cada bloque del notebook
- Ejecutar el modelo y visualizar resultados
- Subir tu versión modificada a tu repositorio de GitHub
- Documentar tus cambios en el `README.md` personal

---

📬 ¿Dudas? Contacta al profesor o deja un issue en el repositorio.


