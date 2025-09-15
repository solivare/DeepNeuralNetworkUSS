import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Cargar configuración desde archivo YAML
def load_config(path="../config.yaml"):
    """
    Carga los parámetros de configuración definidos en un archivo YAML.
    Esto incluye tamaño del batch, número de épocas, estructura del modelo, etc.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Cargar datos preprocesados
def load_data(path="../data/hospital_readmission_clean.csv"):
    """
    Carga el dataset procesado y separa características (X) y variable objetivo (y).
    """
    df = pd.read_csv(path)
    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]
    return X, y

# Normalización y división del dataset
def preprocess(X, y, config):
    """
    1) Split train/test (estratificado).
    2) Identificar columnas continuas en TRAIN y escalar SOLO esas.
    3) Reducir dimensionalidad por varianza en TRAIN y aplicar la misma máscara a TEST.
    4) Convertir a float16 para reducir RAM.
    """
    # --- 1) split primero ---
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y
    )

    # --- 2) columnas continuas (más de 2 valores distintos) en TRAIN ---
    nunique_train = X_train_df.nunique(dropna=False)
    cont_cols = nunique_train[nunique_train > 2].index.tolist()

    # Escalar SOLO continuas
    if len(cont_cols) > 0:
        scaler = StandardScaler()
        X_train_df.loc[:, cont_cols] = scaler.fit_transform(X_train_df[cont_cols])
        X_test_df.loc[:, cont_cols]  = scaler.transform(X_test_df[cont_cols])

    # --- 3) reducción por varianza en TRAIN ---
    # Calculamos varianza de cada columna (ya con continuas escaladas)
    var_train = X_train_df.var(numeric_only=True)
    # Umbral pequeño: elimina columnas casi constantes
    # (ajustable; si sigues con problemas de RAM, sube a 1e-4 o 5e-4)
    keep_mask = var_train > 1e-5
    keep_cols = keep_mask[keep_mask].index.tolist()

    # Aplica la misma selección a TRAIN y TEST
    X_train_df = X_train_df[keep_cols]
    X_test_df  = X_test_df[keep_cols]

    # --- 4) downcast a float16 (mitad de memoria que float32) ---
    X_train_df = X_train_df.astype(np.float16)
    X_test_df  = X_test_df.astype(np.float16)

    # A arrays
    X_train = X_train_df.to_numpy(copy=False)
    X_test  = X_test_df.to_numpy(copy=False)

    y_train = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
    y_test  = y_test.values  if hasattr(y_test, "values")  else np.asarray(y_test)

    return X_train, X_test, y_train, y_test


# Construcción de la red neuronal
def build_model(config, input_dim):
    """
    Red densa con Dropout opcional y L2 (valores desde config.yaml).
    """
    mdl_cfg = config["model"]
    act = mdl_cfg["activation"]
    out_act = mdl_cfg["output_activation"]
    hidden = mdl_cfg["hidden_layers"]
    l2_lambda = float(mdl_cfg.get("l2", 0.001))
    drop_rate = float(mdl_cfg.get("dropout_rate", 0.0))  # 0.0 = sin Dropout

    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for units in hidden:
        model.add(Dense(units, activation=act, kernel_regularizer=l2(l2_lambda)))
        if drop_rate > 0:
            model.add(Dropout(drop_rate))

    model.add(Dense(1, activation=out_act))

    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Entrenamiento del modelo
def train_model(model, X_train, y_train, config):
    """
    Entrena la red. EarlyStopping opcional vía config['early_stopping'].
    """
    es_cfg = config.get("early_stopping", {})
    callbacks = []
    if es_cfg.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.get("monitor", "val_loss"),
                patience=int(es_cfg.get("patience", 3)),
                min_delta=float(es_cfg.get("min_delta", 0.0)),
                restore_best_weights=True
            )
        )

    return model.fit(
        X_train, y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )