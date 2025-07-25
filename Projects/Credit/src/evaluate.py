"""
Evalúa un modelo guardado en models/<RUN_ID>/:
– Carga automáticamente .keras  (Keras/TF)  ó .txt (LightGBM)
– Genera métricas, figuras y texto en reports/<RUN_ID>/
– Busca el umbral que maximiza F1 y lo registra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from joblib import load
import tensorflow as tf                 # para modelos Keras
from lightgbm import Booster            # CHANGE v3: para LightGBM
from numpy import arange

from .config import MODELS_DIR, REPORTS_DIR, RUN_ID
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score
)

# ─── Funciones auxiliares ────────────────────────────────────────
def evaluate_model(y_true, y_pred, y_proba=None, model_name="Modelo"):
    """Imprime y devuelve métricas principales."""
    print(f"\n📊 Evaluación del modelo: {model_name}")
    print(classification_report(y_true, y_pred))

    f1  = f1_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_proba) if y_proba is not None else None

    print(f"F1 Score  : {f1:.4f}")
    print(f"Precision : {pre:.4f}")
    print(f"Recall    : {rec:.4f}")
    if roc is not None:
        print(f"ROC AUC   : {roc:.4f}")

    return {"f1": f1, "precision": pre, "recall": rec, "roc_auc": roc}

def plot_confusion_matrix(y_true, y_pred, path, model_name="Modelo"):
    plt.figure(figsize=(4, 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matriz de Confusión – {model_name}")
    plt.xlabel("Predicción"); plt.ylabel("Real")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

def plot_roc_curve(y_true, y_proba, path, label="Modelo"):
    plt.figure()
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Curva ROC")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

def plot_pr_curve(y_true, y_proba, path, label="Modelo"):
    plt.figure()
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auc_val = auc(recall, precision)
    plt.plot(recall, precision, label=f"{label} (AUC={auc_val:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

def predict_with_threshold(y_proba, threshold=0.5):
    return (y_proba >= threshold).astype(int)

# ─── Entrada principal ───────────────────────────────────────────
if __name__ == "__main__":
    print(f"🔎 RUN_ID = {RUN_ID}")

    # --- cargar modelo (Keras .keras ó LightGBM .txt) ---------------
    keras_file = MODELS_DIR / f"{RUN_ID}_model.keras"
    lgbm_file  = MODELS_DIR / f"{RUN_ID}_model.txt"     # CHANGE v3

    if keras_file.exists():                            # red neuronal
        model = tf.keras.models.load_model(keras_file)
        predict_proba = lambda X: model.predict(X).ravel()
        model_label   = f"NN {RUN_ID}"
    elif lgbm_file.exists():                           # LightGBM
        model = Booster(model_file=str(lgbm_file))
        predict_proba = lambda X: model.predict(X)
        model_label   = f"LGBM {RUN_ID}"
    else:
        raise FileNotFoundError("No se encontró modelo .keras ni .txt")

    # --- cargar hold-out ------------------------------------------
    holdout = load(MODELS_DIR / f"{RUN_ID}_holdout.joblib")
    X_test, y_test = holdout["X_test"], holdout["y_test"]

    # --- predicciones --------------------------------------------
    y_proba = predict_proba(X_test)
    y_pred  = predict_with_threshold(y_proba, 0.5)

    metrics = evaluate_model(y_test, y_pred, y_proba, model_name=model_label)

    # --- threshold óptimo (max F1)  -------------------------------  # CHANGE v3
    best_f1, best_t = 0, 0
    for t in arange(0.2, 0.9, 0.02):
        f1 = f1_score(y_test, predict_with_threshold(y_proba, t))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"⭐ Threshold óptimo={best_t:.2f} → F1={best_f1:.3f}")

    # --- guardar reporte texto ------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))
        f.write(f"\nROC AUC : {metrics['roc_auc']:.4f}\n")
        f.write(f"BestThr : {best_t:.2f}  F1={best_f1:.3f}\n")

    # --- gráficas --------------------------------------------------
    sns.set_theme()
    plot_confusion_matrix(y_test, y_pred,
                          REPORTS_DIR / "confusion_matrix.png",
                          model_name=model_label)

    plot_roc_curve(y_test, y_proba,
                   REPORTS_DIR / "roc_curve.png",
                   label=model_label)

    plot_pr_curve(y_test, y_proba,
                  REPORTS_DIR / "pr_curve.png",
                  label=model_label)

    print("✔ Reportes guardados en", REPORTS_DIR)