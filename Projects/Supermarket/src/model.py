# src/model.py

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization
)
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _resolve_optimizer(opt_cfg):
    """
    Permite usar un string del config (p.ej. "adam")
    o un dict con parámetros ({"name":"adam","lr":1e-3}).
    """
    if isinstance(opt_cfg, str):
        return opt_cfg  # Keras resolverá "adam", "sgd", etc.
    if isinstance(opt_cfg, dict):
        name = opt_cfg.get("name", "adam").lower()
        lr = opt_cfg.get("lr", opt_cfg.get("learning_rate", 1e-3))
        if name == "adam":
            return Adam(learning_rate=lr)
        return Adam(learning_rate=lr)
    return "adam"

# 🧠 Versión mejorada de la CNN
def build_model(config, num_classes):
    input_shape = (
        config["image"]["height"],
        config["image"]["width"],
        config["image"].get("channels", 3)
    )
    l2 = regularizers.l2(1e-4)  # regularización L2 ligera

    model = models.Sequential([
        # Bloque 1
        Conv2D(32, (3,3), activation="relu", padding="same",
               kernel_regularizer=l2, input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.2),

        # Bloque 2
        Conv2D(64, (3,3), activation="relu", padding="same", kernel_regularizer=l2),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.3),

        # Bloque 3
        Conv2D(128, (3,3), activation="relu", padding="same", kernel_regularizer=l2),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.4),

        # Cabeza
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])

    optimizer = _resolve_optimizer(config.get("training", {}).get("optimizer", "adam"))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

    
def build_model_transfer(config, num_classes):
    input_shape = (config["image"]["height"], config["image"]["width"], config["image"].get("channels", 3))
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False  # Fase 1: congelado

    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model
    
