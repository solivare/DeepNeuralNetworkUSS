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

# Este modelo es muy simple. Puedes mejorar arquitectura,
# agregar regularización, más bloques, batch normalization, etc.
def build_model(config, num_classes):
    input_shape = (
        config["image"]["height"],
        config["image"]["width"],
        config["image"]["channels"],
    )

    model = models.Sequential([
        # Primer bloque convolucional
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Segundo bloque convolucional
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Tercer bloque convolucional
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Capas densas (Fully Connected)
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # Capa de salida
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=config["training"]["optimizer"],
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

