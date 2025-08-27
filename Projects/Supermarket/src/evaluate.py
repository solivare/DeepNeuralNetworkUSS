import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical

def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False):
    """
    Matriz de confusión mejorada para muchos labels.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    plt.figure(figsize=(18, 16))  # Ajusta según tus clases

    sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names, 
                square=True, cbar=True)

    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.title("Matriz de Confusión" + (" (Normalizada)" if normalize else ""))
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def show_classification_report(y_true, y_pred, class_names):
    """
    Imprime el classification report
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("📋 Reporte de Clasificación:\n")
    print(report)


def plot_prediction_distribution(y_true, y_prob, class_index=0, class_name=""):
    """
    Dibuja la distribución de probabilidades para una clase específica en multiclase.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    plt.figure(figsize=(6, 4))
    plt.hist(y_prob[y_true == class_index], bins=30, alpha=0.6, label=f"{class_name} (verdadera)", color="skyblue", density=True)
    plt.hist(y_prob[y_true != class_index], bins=30, alpha=0.6, label="Otras clases", color="salmon", density=True)
    plt.axvline(0.5, color="gray", linestyle="--", label="Umbral 0.5")
    plt.title(f"Distribución de probabilidades - Clase: {class_name}")
    plt.xlabel("Probabilidad predicha")
    plt.ylabel("Densidad")
    plt.legend()
    plt.grid(True)
    plt.show()


def predict_classes(model, generator):
    """
    Predice las clases sobre un generador (test o validación)
    """
    y_prob = model.predict(generator)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = generator.classes
    class_names = list(generator.class_indices.keys())
    return y_true, y_pred, y_prob, class_names