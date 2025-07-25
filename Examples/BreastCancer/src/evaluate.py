import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2

def plot_confusion_matrix(y_true, y_pred, labels=["Benigno", "Maligno"]):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.show()

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend()
    plt.grid()
    plt.show()

#def plot_prediction_distribution(y_true, y_scores):
#    plt.figure(figsize=(6,4))
#    sns.histplot(y_scores[y_true == 0], label="Clase 0", color="skyblue", kde=True)
#    sns.histplot(y_scores[y_true == 1], label="Clase 1", color="salmon", kde=True)
#    plt.xlabel("Probabilidad predicha")
#    plt.ylabel("Frecuencia")
#    plt.title("Distribución de probabilidades")
#    plt.legend()
#    plt.grid()
#    plt.show()

def plot_prediction_distribution(y_true, y_prob):
    """
    Grafica la distribución de probabilidades para cada clase
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob).flatten()

    plt.figure(figsize=(8, 4))
    plt.hist(y_prob[y_true == 0], bins=30, alpha=0.6, label="Benigno", color="skyblue", density=True)
    plt.hist(y_prob[y_true == 1], bins=30, alpha=0.6, label="Maligno", color="salmon", density=True)
    plt.axvline(0.5, color="gray", linestyle="--", label="Umbral 0.5")
    plt.xlabel("Probabilidad Predicha")
    plt.ylabel("Densidad")
    plt.title("Distribución de Probabilidades por Clase")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_f1_vs_threshold(y_true, y_scores, modelo=""):
    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores = []
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)

    plt.figure(figsize=(6,4))
    plt.plot(thresholds, f1_scores, marker='o')
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title(f"F1 Score vs Threshold ({modelo})")
    plt.grid()
    plt.show()

def show_misclassified_images(generator, model, class_names=["Benigno", "Maligno"], max_images=12):
    """
    Muestra imágenes mal clasificadas por la red neuronal.

    Parameters:
        generator: ImageDataGenerator (test or validation)
        model: modelo Keras entrenado
        class_names: nombres de las clases
        max_images: número máximo de errores a mostrar
    """
    print("🔍 Buscando errores de clasificación...")

    # Obtener imágenes y etiquetas verdaderas
    x_batch, y_true = next(generator)
    y_pred = model.predict(x_batch).ravel()
    y_pred_label = (y_pred > 0.5).astype(int)

    # Índices donde hay error
    misclassified = np.where(y_pred_label != y_true)[0]

    if len(misclassified) == 0:
        print("✅ ¡No hay errores en este batch!")
        return

    # Mostrar imágenes mal clasificadas
    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(misclassified[:max_images]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(x_batch[idx])
        plt.axis("off")
        true_lbl = class_names[int(y_true[idx])]
        pred_lbl = class_names[int(y_pred_label[idx])]
        plt.title(f"Real: {true_lbl}\nPred: {pred_lbl} ({y_pred[idx]:.2f})")
    plt.tight_layout()
    plt.show()


def generate_gradcam(model, img_array, class_idx=None, layer_name=None):
    """
    Genera el mapa Grad-CAM para una imagen y modelo dado.

    Args:
        model: modelo Keras entrenado (Sequential).
        img_array: imagen con shape (1, height, width, channels).
        class_idx: índice de la clase objetivo (por defecto se toma la clase predicha).
        layer_name: nombre de la última capa convolucional (opcional).

    Returns:
        heatmap: mapa de activación Grad-CAM.
    """

    # 🔧 1. Determinar clase predicha si no se entrega
    if class_idx is None:
        preds = model.predict(img_array)
        class_idx = int(preds[0] > 0.5)  # Para binaria

    # 🔍 2. Identificar capa convolucional si no se especifica
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

    if layer_name is None:
        raise ValueError("No se encontró una capa Conv2D en el modelo.")

    # 🧠 Crear modelo Grad-CAM desde entrada concreta
    #grad_model = Model(
    #    inputs=[img_array],
    #    outputs=[model.get_layer(layer_name).output, model(img_array)]
    #)

    # 🧠 Crear modelo Grad-CAM correctamente
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # Solo una salida binaria

    # 📉 4. Gradientes de la clase respecto a la salida de la capa convolucional
    grads = tape.gradient(loss, conv_outputs)

    # 🧮 5. Promedio espacial de los gradientes
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 📸 6. Multiplicar por activaciones y promediar
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 🔥 7. Normalizar entre 0 y 1
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-10

    return heatmap.numpy()

def display_gradcam(image_array, heatmap, alpha=0.4):
    """
    Muestra la imagen original sobrepuesta con el heatmap
    """
    img = np.uint8(255 * image_array[0])
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.show()