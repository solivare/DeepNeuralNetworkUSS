# 🧠 Comparación de Modelos - Clasificación Histopatológica

Este documento presenta una comparación entre dos versiones de una red neuronal para clasificar imágenes histopatológicas en benignas o malignas.

---

## 📌 Cambios en la Nueva Versión

Se realizaron las siguientes mejoras en el modelo original:

1. 🔁 **BatchNormalization**: aplicado después de cada capa convolucional.
2. 🎯 **Dropout (0.5)**: antes de la capa de salida para evitar overfitting.
3. 🧠 **Arquitectura más profunda**: se añadió una tercera capa convolucional.
4. ⚙️ **Learning rate reducido**: se ajustó a `0.0005` para una convergencia más estable.

---

## 📊 Comparación de Resultados

| Métrica         | Modelo Original | Modelo Mejorado |
|-----------------|------------------|------------------|
| Accuracy        | 0.8288           | **0.8575**       |
| F1-score macro  | 0.83             | **0.86**         |
| Pérdida (loss)  | 0.4280           | **0.3936**       |
| TP Maligno      | 327              | **337**          |
| FP Maligno      | 64               | **51**           |
| FN Maligno      | 73               | **63**           |

---

## 🖼️ Visualizaciones Generadas

- 📉 Curva F1 vs Threshold
- 📊 Distribución de probabilidades por clase
- ❌ Imágenes mal clasificadas por la red neuronal

---

## 🧪 Conclusión

Las modificaciones realizadas mejoraron la precisión general, el balance entre precisión y recall (F1-score) y la pérdida del modelo. La arquitectura mejorada captura mejor los patrones relevantes en las imágenes sin caer en overfitting.

