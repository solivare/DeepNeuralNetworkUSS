import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def evaluate_model(y_true, y_pred, y_proba=None, model_name="Modelo"):
    print(f"\n📊 Evaluación del modelo: {model_name}")
    print(classification_report(y_true, y_pred))

    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_proba) if y_proba is not None else None

    print(f"F1 Score     : {f1:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    if roc is not None:
        print(f"ROC AUC      : {roc:.4f}")

    return {"f1": f1, "precision": precision, "recall": recall, "roc_auc": roc}

def plot_confusion_matrix(y_true, y_pred, model_name="Modelo"):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matriz de Confusión - {model_name}")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

def plot_roc_comparison(y_true_1, y_proba_1, y_true_2=None, y_proba_2=None,
                        label1="Modelo 1", label2="Modelo 2"):
    fpr1, tpr1, _ = roc_curve(y_true_1, y_proba_1)
    auc1 = auc(fpr1, tpr1)
    plt.plot(fpr1, tpr1, label=f"{label1} (AUC={auc1:.2f})")

    if y_true_2 is not None and y_proba_2 is not None:
        fpr2, tpr2, _ = roc_curve(y_true_2, y_proba_2)
        auc2 = auc(fpr2, tpr2)
        plt.plot(fpr2, tpr2, label=f"{label2} (AUC={auc2:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label="Aleatorio")
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title("Comparación de Curvas ROC")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_precision_recall_curves(y_true_1, y_proba_1, y_true_2=None, y_proba_2=None,
                                 label1="Modelo 1", label2="Modelo 2"):
    precision1, recall1, _ = precision_recall_curve(y_true_1, y_proba_1)
    auc1 = auc(recall1, precision1)
    plt.plot(recall1, precision1, label=f"{label1} (AUC={auc1:.2f})")

    if y_true_2 is not None and y_proba_2 is not None:
        precision2, recall2, _ = precision_recall_curve(y_true_2, y_proba_2)
        auc2 = auc(recall2, precision2)
        plt.plot(recall2, precision2, label=f"{label2} (AUC={auc2:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curvas Precision-Recall Comparadas")
    plt.legend()
    plt.grid(True)
    plt.show()

def predict_with_threshold(y_proba, threshold=0.5):
    return (y_proba >= threshold).astype(int)

def evaluate_thresholds(y_true, y_proba, thresholds=np.arange(0.1, 1.0, 0.1)):
    results = {
        "Threshold": [],
        "Precision": [],
        "Recall": [],
        "F1-score": []
    }

    for t in thresholds:
        y_pred = predict_with_threshold(y_proba, threshold=t)
        results["Threshold"].append(round(t, 2))
        results["Precision"].append(precision_score(y_true, y_pred, zero_division=0))
        results["Recall"].append(recall_score(y_true, y_pred, zero_division=0))
        results["F1-score"].append(f1_score(y_true, y_pred, zero_division=0))

    return pd.DataFrame(results)

def compare_models_metrics(y_true_1, y_pred_1, y_proba_1,
                           y_true_2, y_pred_2, y_proba_2,
                           model_name_1="Modelo 1", model_name_2="Modelo 2"):
    results = {
        "Modelo": [],
        "Precision": [],
        "Recall": [],
        "F1-score": [],
        "ROC AUC": []
    }

    # Modelo 1
    results["Modelo"].append(model_name_1)
    results["Precision"].append(precision_score(y_true_1, y_pred_1, zero_division=0))
    results["Recall"].append(recall_score(y_true_1, y_pred_1, zero_division=0))
    results["F1-score"].append(f1_score(y_true_1, y_pred_1, zero_division=0))
    results["ROC AUC"].append(roc_auc_score(y_true_1, y_proba_1))

    # Modelo 2
    results["Modelo"].append(model_name_2)
    results["Precision"].append(precision_score(y_true_2, y_pred_2, zero_division=0))
    results["Recall"].append(recall_score(y_true_2, y_pred_2, zero_division=0))
    results["F1-score"].append(f1_score(y_true_2, y_pred_2, zero_division=0))
    results["ROC AUC"].append(roc_auc_score(y_true_2, y_proba_2))

    return pd.DataFrame(results)