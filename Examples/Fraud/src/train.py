import pandas as pd
import yaml
import os
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import numpy as np

def load_config(config_filename="config.yaml"):
    base_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_path, ".."))
    config_path = os.path.join(root_dir, config_filename)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["base_dir"] = root_dir  # Esto permite rutas absolutas
    return config

def load_data(path, base_dir=""):
    full_path = os.path.join(base_dir, path)
    df = pd.read_csv(full_path)
    X = df.drop("Class", axis=1).values
    y = df["Class"].values
    return X, y

#def build_model(input_dim, config):
#    model = Sequential()
#    model.add(Dense(config["model"]["units1"], activation="relu", input_dim=input_dim))
#    model.add(Dropout(config["model"]["dropout"]))
#    model.add(Dense(config["model"]["units2"], activation="relu"))
#    model.add(Dropout(config["model"]["dropout"]))
#    model.add(Dense(1, activation="sigmoid"))
#
#    model.compile(
#        optimizer=config["model"]["optimizer"],
#        loss="binary_crossentropy",
#        metrics=["accuracy"]
#    )
#    return model

def build_model(input_dim, config):
    model_type = config["model"].get("model_type", "deep")
    dropout = config["model"].get("dropout", 0.3)
    l2_reg = l2(config["model"].get("l2", 0.0))

    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    if model_type == "simple":
        model.add(Dense(16, activation="relu", kernel_regularizer=l2_reg))
        model.add(Dropout(dropout))
    else:  # deep
        model.add(Dense(config["model"]["units1"], activation="relu", kernel_regularizer=l2_reg))
        model.add(Dropout(dropout))
        model.add(Dense(config["model"]["units2"], activation="relu", kernel_regularizer=l2_reg))
        model.add(Dropout(dropout))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=config["model"].get("optimizer", "adam"),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

def train(config_path="config.yaml", return_history=False):
    config = load_config(config_path)

    # Usar train_full.csv si se activa use_class_weight, si no usar train.csv
    train_path = (
        config["paths"].get("train_full")
        if config.get("training", {}).get("use_class_weight", False)
        else config["paths"]["train"]
    )

    if train_path is None:
        raise ValueError("🚨 No se encontró 'train_full' en config['paths']. Verifica config.yaml.")

    # Cargar datos
    X_train, y_train = load_data(train_path, config["base_dir"])
    X_val, y_val = load_data(config["paths"]["val"], config["base_dir"])

    # Escalar
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Crear modelo
    model = build_model(X_train.shape[1], config)

    # class_weight opcional
    class_weight = None
    if config["training"].get("use_class_weight", False):
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight = dict(enumerate(weights))
        print("✅ Usando class_weight:", class_weight)

    # Entrenamiento
    callbacks = []
    early_cfg = config["training"].get("early_stopping", {})
    if early_cfg.get("enabled", False):
        patience = early_cfg.get("patience", 5)
        print(f"🛑 EarlyStopping ACTIVADO (patience = {patience})")
        callbacks.append(EarlyStopping(patience=patience, restore_best_weights=True))
    else:
        print("🔁 EarlyStopping DESACTIVADO (entrenará todas las épocas)")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        callbacks=callbacks,
        class_weight=class_weight
    )

    # Guardar modelo
    model_dir = os.path.join(config["base_dir"], "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.keras")  # ✅ usa el nuevo formato

    model.save(model_path)
    print(f"✅ Modelo guardado en {model_path}")

    if return_history:
        return history

if __name__ == "__main__":
    train()