import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, save_model

def plot_training_history(history, model_name="Modelo"):
    """
    Grafica la historia de entrenamiento (loss y accuracy)
    """
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "bo-", label="Entrenamiento")
    plt.plot(epochs, val_acc, "ro-", label="Validación")
    plt.title(f"{model_name} - Accuracy")
    plt.xlabel("Épocas")
    plt.ylabel("Precisión")
    plt.legend()
    plt.grid()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "bo-", label="Entrenamiento")
    plt.plot(epochs, val_loss, "ro-", label="Validación")
    plt.title(f"{model_name} - Pérdida")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def save_model(model, path="models/model.keras"):
    """
    Guarda el modelo entrenado en la ruta especificada
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"✅ Modelo guardado en: {path}")


def load_model_from_file(path="models/model.keras"):
    """
    Carga un modelo entrenado desde archivo
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo del modelo: {path}")
    return load_model(path)