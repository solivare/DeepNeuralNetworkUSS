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

## Cambios ejecutados al archivo main.ipynb y config.yaml para mejorar el modelo base

### Balanceo de clases:
Debido a las diferencias en los soportes de las clases, se implementó "class_weight" de la siguiente forma:

weights = class_weight.compute_class_weight(class_weight='balanced',
                                            classes=np.unique(y_train),
                                            y=y_train)

class_weights = dict(zip(np.unique(y_train), weights))

Antes de aplicar balanceo el Accuracy del modelo era de 87% y la precisión del 92,6%. Sin embargo la métrica más relevante del modelo (Recall) era del 89%. Sin embargo, luego de aplicar el balanceo de clases, y luego de varias iteraciones en el archivo config.yaml, no se logró mejorar el Recall de la clase 1, el cual es la mas importante del modelo (Un cliente que pude haber retenido.). La máxima encontrada fué 86% con los siguientes parámetros:

model:
  hidden_units: [64, 32, 16]       
  activation: tanh            
  output_activation: sigmoid  

training:
  batch_size: 32              
  epochs: 100                 
  optimizer: adam             
  loss: binary_crossentropy   


