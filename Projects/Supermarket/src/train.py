import os
import yaml
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_model


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_data_generators(config):
    """
    Crea generadores de imágenes para entrenamiento y validación
    """
    data_dir = config["paths"]["prepared_data"]

    # Aumentación solo para entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    target_size = (
        config["image"]["height"],
        config["image"]["width"]
    )
    batch_size = config["training"]["batch_size"]

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False # Esto asegura que el orden de las imágenes sea fijo y reproducible cualquier función que compare predidicción y etiqueta
    )

    return train_generator, val_generator


def train_model(config):
    train_gen, val_gen = get_data_generators(config)

    num_classes = train_gen.num_classes
    model = build_model(config, num_classes)

    callbacks = [
        EarlyStopping(patience=config["training"]["patience"], restore_best_weights=True),
        ModelCheckpoint(config["paths"]["model"], save_best_only=True)
    ]

    history = model.fit(
        train_gen,
        epochs=config["training"]["epochs"],
        validation_data=val_gen,
        callbacks=callbacks
    )

    return model, history, train_gen.class_indices