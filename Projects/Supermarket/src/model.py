# src/model.py

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ⚠️ Este modelo es muy simple. Puedes mejorar arquitectura,
# agregar regularización, más bloques, batch normalization, etc.
def build_model(config, num_classes):
    input_shape = (
        config["image"]["height"],
        config["image"]["width"],
        config["image"]["channels"]
    )

    model = models.Sequential()

    # Bloque 1
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # Bloque 2
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    # ✅ AGREGAR CAPAS FALTANTES PARA CLASIFICACIÓN
    # Aplanar la salida convolucional
    model.add(layers.Flatten())
    
    # Capa densa oculta
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    
    # Capa de salida con el número correcto de clases
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=config["training"]["optimizer"],
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model