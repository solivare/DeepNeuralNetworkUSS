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

# --- imports necesarios (si no est谩n ya en el archivo) ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def plot_f1_vs_threshold(y_true_bin, y_scores, modelo="CNN"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score
    import numpy as np

    thresholds = np.linspace(0.0, 1.0, 101)
    f1s = []

    for thr in thresholds:
        y_pred_bin = (y_scores >= thr).astype(int)
        f1 = f1_score(y_true_bin, y_pred_bin)
        f1s.append(f1)

    best_idx = np.argmax(f1s)
    best_thr = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    plt.plot(thresholds, f1s, label="F1-score")
    plt.axvline(best_thr, color="red", linestyle="--", label=f"Best threshold = {best_thr:.2f}")
    plt.title(f"F1 vs Threshold - Modelo {modelo}")
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.legend()
    plt.grid()
    plt.show()

    return best_thr, best_f1


def show_misclassified_images(generator, model, class_names, n=12):
    
    # Asegurar
    generator.reset()

    # Probabilidades y predicciones
    y_prob = model.predict(generator, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = generator.classes

    # 脥ndices mal clasificados
    wrong_idx = np.where(y_pred != y_true)[0]
    if len(wrong_idx) == 0:
        print("No hay imagen mal clasificadas.")
        return

    # Tomar hasta n 
    idx = wrong_idx[:n]

    # Cargar y dibujar
    import math
    import matplotlib.image as mpimg

    cols = 4
    rows = math.ceil(len(idx) / cols)
    plt.figure(figsize=(cols * 4, rows * 4))

    for i, k in enumerate(idx):
        img_path = generator.filepaths[k]
        img = mpimg.imread(img_path)
        true_lbl = class_names[y_true[k]]
        pred_lbl = class_names[y_pred[k]]
        pred_p   = np.max(y_prob[k])

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.set_title(f"T:{true_lbl}\nP:{pred_lbl} ({pred_p:.2f})", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

