"""
Entrena LightGBM sobre credit_clean.csv y genera artefactos v3_lgbm
# CHANGE v3: nuevo modelo tabular clásico, suele superar NN en AUC.
"""

import pandas as pd
import yaml
from pathlib import Path
from joblib import dump
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier           # pip install lightgbm
from sklearn.metrics import roc_auc_score
from src.config import BASE_DIR, DATA_DIR, MODELS_DIR, RUN_ID

# ─── utilidades ──────────────────────────────────────────────────
def load_config(path: Path = BASE_DIR / "config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def load_data(path: Path = DATA_DIR / "credit_clean.csv"):
    df = pd.read_csv(path)
    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]
    return X, y

# ─── main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config()
    X, y = load_data()

    print(f"🔎 RUN_ID = {RUN_ID}  (LightGBM)")

    # división estratificada (igual que NN)
    Xtr, Xts, ytr, yts = train_test_split(
        X, y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y
    )

    # pesos inverso-frecuencia para clase minoritaria
    scale_pos_weight = (len(ytr) - ytr.sum()) / ytr.sum()

    # construir modelo con hiperparámetros del YAML
    lgbm_params = cfg["lgbm"]
    model = LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,     # balance automático
        **lgbm_params,
        random_state=cfg["random_state"]
    )

    model.fit(Xtr, ytr)
    auc = roc_auc_score(yts, model.predict_proba(Xts)[:, 1])
    print(f"✔ ROC-AUC hold-out: {auc:.4f}")

    # ── guardar artefactos ──────────────────────────────────────
    model_path   = MODELS_DIR / f"{RUN_ID}_model.txt"
    holdout_path = MODELS_DIR / f"{RUN_ID}_holdout.joblib"

    model.booster_.save_model(str(model_path))
    dump({"X_test": Xts, "y_test": yts}, holdout_path)

    print(f"✔ Modelo LightGBM guardado → {model_path}")
    print(f"✔ Hold-out guardado        → {holdout_path}")