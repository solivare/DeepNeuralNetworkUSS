"""
Entrenamiento del experimento identificado por RUN_ID
– Añade class_weight 'balanced' para la clase minoritaria
– Implementa Early-Stopping (patience configurable)
– Guarda scaler y artefactos en models/<run_id>/
"""

import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
from tensorflow.keras.callbacks import EarlyStopping          # CHANGE v2
from src.config import BASE_DIR, DATA_DIR, MODELS_DIR, RUN_ID
from .model  import build_model


# ─── utilidades de carga ─────────────────────────────────────────
def load_config(path: Path = BASE_DIR / "config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def load_data(path: Path = DATA_DIR / "credit_clean.csv"):
    df = pd.read_csv(path)
    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]
    return X, y


# ─── pre-procesado (escalado + split) ───────────────────────────
def preprocess(X, y, cfg):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y
    )

    # Guarda scaler con el run_id para reproducir later
    dump(scaler, MODELS_DIR / f"{RUN_ID}_scaler.joblib")      # CHANGE v2
    return X_train, X_test, y_train, y_test


# ─── entrenamiento con mejoras v2 ───────────────────────────────
def train_model(model, X_train, y_train, cfg):
    """
    • Early-Stopping (si está definido en YAML).
    • Ponderación automática de clases cuando class_weight == 'balanced'.
    """
    callbacks = []
    if "early_stopping" in cfg:                               # CHANGE v2
        callbacks.append(
            EarlyStopping(
                patience=cfg["early_stopping"]["patience"],
                restore_best_weights=True,
                verbose=1
            )
        )

    class_w = None                                           # CHANGE v2
    if cfg.get("class_weight") == "balanced":
        pos = y_train.sum()
        neg = len(y_train) - pos
        class_w = {0: 1, 1: neg / pos}   # inverso de la frecuencia

    history = model.fit(
        X_train, y_train,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        validation_split=0.20,
        class_weight=class_w,            # CHANGE v2
        callbacks=callbacks,
        verbose=1
    )
    return history


# ─── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config()

    X, y = load_data()
    print(f"🔎 RUN_ID = {RUN_ID}")
    print("Distribución target:\n", y.value_counts(normalize=True))

    Xtr, Xts, ytr, yts = preprocess(X, y, cfg)

    model   = build_model(cfg, Xtr.shape[1])                  # usa nueva red
    history = train_model(model, Xtr, ytr, cfg)

    model_path   = MODELS_DIR / f"{RUN_ID}_model.keras"
    holdout_path = MODELS_DIR / f"{RUN_ID}_holdout.joblib"

    model.save(model_path)
    dump({"X_test": Xts, "y_test": yts}, holdout_path)
    dump({"history": history.history},
         MODELS_DIR / f"{RUN_ID}_history.joblib")

    print(f"✔ Modelo guardado   → {model_path}")
    print(f"✔ Hold-out guardado → {holdout_path}")