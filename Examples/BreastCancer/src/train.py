import os
import yaml
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_model

def load_config(path="../config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_data_generators(config):
    """
    Crea generadores de imágenes para entrenamiento y validación
    """
    data_dir = config["paths"]["prepared_data"]

    # Preprocesamiento y aumentación para entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        shear_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )


    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=(config["image"]["height"], config["image"]["width"]),
        batch_size=config["training"]["batch_size"],
        class_mode="binary"
    )

    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=(config["image"]["height"], config["image"]["width"]),
        batch_size=config["training"]["batch_size"],
        class_mode="binary",
        shuffle=False
    )

    return train_generator, test_generator

def train_model(config):
    train_gen, test_gen = get_data_generators(config)

    model = build_model(config)

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(config["paths"]["model"], save_best_only=True)
    ]

    history = model.fit(
        train_gen,
        epochs=config["training"]["epochs"],
        validation_data=test_gen,
        callbacks=callbacks
    )

    return model, history

def load_datasets(data_dir, image_size, batch_size, validation_split=0.2, seed=42):
    """
    Carga conjuntos de datos de imágenes desde carpetas usando ImageDataGenerator
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )

    train_ds = train_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        seed=seed
    )

    val_ds = train_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        seed=seed
    )

    return train_ds, val_ds

if __name__ == "__main__":
    config = load_config()
    model, history = train_model(config)