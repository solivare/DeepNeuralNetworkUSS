import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
from tensorflow.keras.models import load_model
import os

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

def evaluate_model(config_path="config.yaml"):
    config = load_config(config_path)

    # Elegir test original o balanceado
    use_balanced_test = config.get("evaluation", {}).get("use_balanced_test", False)
    test_path = config["paths"].get("test_balanced") if use_balanced_test else config["paths"]["test"]

    print(f"📊 Evaluando sobre el set: {'balanceado' if use_balanced_test else 'completo'}")

    # Cargar datos
    X_test, y_test = load_data(test_path, config["base_dir"])
    
    # Escalar
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

    # Cargar modelo
    model_path = os.path.join(config["base_dir"], "models", "model.keras")
    model = load_model(model_path)
    
    # Predicciones
    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = auc(*roc_curve(y_test, y_prob)[:2])

    print("🔎 Resultados:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC:       {roc_auc:.4f}")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Fraude", "Fraude"], yticklabels=["No Fraude", "Fraude"])
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.show()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="darkorange")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title("Curva ROC")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    evaluate_model()