import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def plot_training_history(history, model_name="Red Neuronal"):
    """
    Grafica la evolución del loss y accuracy durante el entrenamiento
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.title(f'Precisión - {model_name}')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.title(f'Pérdida - {model_name}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def save_model(model, path="../models/model.keras"):
    """
    Guarda el modelo entrenado en formato keras
    """
    model.save(path)
    print(f"✅ Modelo guardado en {path}")

def load_saved_model(path="../models/model.keras"):
    """
    Carga un modelo previamente guardado
    """
    return load_model(path)