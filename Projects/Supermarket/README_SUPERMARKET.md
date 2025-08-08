
# 🛒 Clasificación Multiclase de Productos de Supermercado  

## 🎯 Objetivo  
Entrenar un modelo de red neuronal para clasificar imágenes de productos de supermercado en **53 clases** diferentes, optimizando la precisión y la generalización incluso con un dataset reducido y clases desbalanceadas.  

---

## 📂 Dataset y Preprocesamiento  
- Estructura esperada:  
```
data/items_prepared/{train,val,test}/{clase}/imagen.jpg
```  
- Las imágenes se reorganizan por clase en `paths.prepared_data` (definido en `config.yaml`).  
- Resolución utilizada: **128×128** píxeles.  
- Normalización: `preprocess_input` (MobileNetV2) para compatibilidad con pesos preentrenados.  

---

## 📈 Evolución de Métricas

| Versión | Descripción | Accuracy | Macro F1 | Weighted F1 |
|---------|-------------|----------|----------|-------------|
| **v1** | CNN simple + `rescale=1./255` | 0.23 | 0.18 | 0.21 |
| **v2** | CNN simple + `class_weight` | ~0.05 | — | — |
| **v3** | **MobileNetV2 congelada** + `preprocess_input` + Dropout 0.3 | **0.755** | **0.679** | **0.739** |

**Lectura rápida:**  
- El cambio a **transfer learning** con MobileNetV2 fue clave (+52 pts en accuracy).  
- `class_weight` no mejoró la CNN simple debido a la arquitectura limitada y dataset reducido.  
- El modelo final generaliza mejor y mantiene buen balance entre clases.  

---

## ⚙️ Configuración Final  
- **Modelo:** MobileNetV2 (`weights=imagenet`) congelada + GlobalAveragePooling + Dropout 0.3 + Dense Softmax  
- **Input:** 128×128 px, RGB  
- **Batch size:** 32  
- **Callbacks:** EarlyStopping (patience=8), ReduceLROnPlateau, ModelCheckpoint  
- **Balanceo:** sin `class_weight` (la mejora provino del transfer learning)  

---

## 📊 Resultados Finales (Validación)  
| Métrica | Valor |
|---------|-------|
| Accuracy | **0.755** |
| Macro F1 | **0.679** |
| Weighted F1 | **0.739** |

**Hallazgos:**  
- Buen rendimiento global, con mejoras amplias en envases y hortalizas.  
- Persisten 3–5 clases con desempeño bajo (p.ej., algunas variantes de yogur/crema y melón).  
- El modelo es **presentable para entrega y demostración**.  

---

## 🔁 Reproducibilidad  

### 1. Preprocesamiento  
- Definir rutas y parámetros en `config.yaml`:  
```yaml
paths:
  prepared_data: "C:/Users/USS/DeepNeuralNetworkUSS/Projects/Supermarket/data/items_prepared"
  model: "models/cnn_multiclass_v2.keras"
image:
  height: 128
  width: 128
training:
  batch_size: 32
  patience: 8
```

### 2. Entrenamiento (`RedNeuronal.ipynb`)  
```python
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.1, zoom_range=0.15, horizontal_flip=True, fill_mode="nearest"
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
cnn_model = model.build_model_transfer(config, num_classes)
cnn_model.fit(train_generator, validation_data=val_generator, callbacks=callbacks)
```

### 3. Evaluación (`Evaluacion.ipynb`)  
- Cargar modelo desde `paths.model`  
- Calcular `accuracy`, `macro-F1`, `weighted-F1`  
- Generar matriz de confusión y reporte de clasificación  
- Guardar métricas en `results/metrics_v3.yaml` (opcional)  

---

## 📌 Limitaciones  
- Variabilidad de iluminación/envase afecta algunas clases.  
- Dataset reducido en ciertas clases (<6 imágenes en train) → recall bajo.  

---

## 🧭 Próximos pasos (opcionales)  
1. **Fine-tuning parcial** del backbone (últimas ~40 capas, LR=1e-4).  
2. Aumentar resolución a **160×160** si el hardware lo permite.  
3. Augmentations dirigidas (brillo, contraste, zoom) en clases problemáticas.  
4. Oversampling ligero en clases con muy pocos ejemplos.  
