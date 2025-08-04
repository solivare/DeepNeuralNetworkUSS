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

    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=config["training"]["optimizer"],
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model