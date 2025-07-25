from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ks_2samp

def plot_training(history):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
    ax1.plot(history.history['loss'], label='train_loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_title("Pérdida por época")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Pérdida")
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='train_acc')
    ax2.plot(history.history['val_accuracy'], label='val_acc')
    ax2.set_title("Precisión por época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Precisión")
    ax2.legend()

    plt.tight_layout()
    plt.show()

def get_training_data_path(config):
    """
    Devuelve el path correcto del dataset de entrenamiento según si se usa class_weight o no.

    Parámetros:
    -----------
    config : dict
        Configuración cargada desde config.yaml.

    Retorna:
    --------
    str : path al archivo CSV que debe usarse como entrenamiento
    """
    use_weights = config.get("training", {}).get("use_class_weight", False)
    if use_weights:
        return config["paths"].get("train_full")
    else:
        return config["paths"]["train"]
    
def plot_confusion_matrix(y_true, y_pred, model_name="Modelo", cmap="Blues"):
    """
    Muestra la matriz de confusión de un modelo.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Fraude", "Fraude"])
    disp.plot(cmap=cmap)
    plt.title(f"Matriz de Confusión - {model_name}")
    plt.grid(False)
    plt.show()

def get_classification_metrics_df(y_true_log, y_pred_log, y_prob_log, 
                                  y_true_nn, y_pred_nn, y_prob_nn):
    """
    Devuelve un DataFrame con métricas de clasificación para dos modelos.
    """
    metrics = {
        "Modelo": ["Regresión Logística", "Red Neuronal"],
        "Accuracy": [
            accuracy_score(y_true_log, y_pred_log),
            accuracy_score(y_true_nn, y_pred_nn)
        ],
        "Precision": [
            precision_score(y_true_log, y_pred_log),
            precision_score(y_true_nn, y_pred_nn)
        ],
        "Recall": [
            recall_score(y_true_log, y_pred_log),
            recall_score(y_true_nn, y_pred_nn)
        ],
        "F1-score": [
            f1_score(y_true_log, y_pred_log),
            f1_score(y_true_nn, y_pred_nn)
        ],
        "AUC": [
            roc_auc_score(y_true_log, y_prob_log),
            roc_auc_score(y_true_nn, y_prob_nn)
        ]
    }
    df = pd.DataFrame(metrics)
    return df.set_index("Modelo").round(4)

def plot_f1_vs_threshold(y_true, y_probs, modelo="Modelo"):
    thresholds = np.arange(0.0, 1.01, 0.01)
    f1_scores = [f1_score(y_true, (y_probs > t).astype(int)) for t in thresholds]

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_scores, label="F1-score")
    plt.axvline(x=best_threshold, color='r', linestyle='--', label=f"Umbral óptimo = {best_threshold:.2f}")
    plt.title(f"F1-score vs Threshold - {modelo}")
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"🔎 Mejor threshold para {modelo}: {best_threshold:.2f} con F1-score = {best_f1:.4f}")

def plot_prediction_distribution(y_true, y_prob, log_scale=False):
    """
    Grafica la distribución de las probabilidades predichas para cada clase.
    
    Args:
        y_true: etiquetas verdaderas (0 o 1)
        y_prob: probabilidades predichas (de un modelo)
        log_scale: si True, muestra eje y en escala logarítmica
    """
    df = pd.DataFrame({
        "Probabilidad": y_prob,
        "Clase": ["No Fraude" if y == 0 else "Fraude" for y in y_true]
    })

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="Probabilidad", hue="Clase", bins=50, stat="density", common_norm=False, alpha=0.5)
    plt.axvline(0.5, color="red", linestyle="--", label="Threshold 0.5")
    plt.xlabel("Probabilidad predicha")
    plt.ylabel("Densidad")
    plt.title("Distribución de probabilidades por clase")
    if log_scale:
        plt.yscale("log")
        plt.ylabel("Densidad (escala log)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_ks_overtraining(y_train, y_prob_train, y_test, y_prob_test, bins=40):
    """
    Visualiza la distribución de predicciones (scores) para señal y background
    en train y test. Incluye test de Kolmogorov-Smirnov para comparar.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scipy.stats import ks_2samp

    # Crear dataframes
    df_train = pd.DataFrame({
        "score": y_prob_train,
        "label": y_train,
        "set": "Train"
    })
    df_test = pd.DataFrame({
        "score": y_prob_test,
        "label": y_test,
        "set": "Test"
    })
    df = pd.concat([df_train, df_test], ignore_index=True)
    df["class"] = df["label"].map({0: "Background", 1: "Señal"})

    # Estética seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Paleta
    palette = {
        "Señal": "#1f77b4",
        "Background": "#d62728"
    }

    for clase in ["Señal", "Background"]:
        # Train como barra
        subset_train = df[(df["class"] == clase) & (df["set"] == "Train")]
        counts_train, bin_edges = np.histogram(
            subset_train["score"], bins=bins, range=(0, 1), density=True
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.bar(
            bin_centers, counts_train, width=(1 / bins),
            label=f"{clase} (Train)", alpha=0.4, color=palette[clase], edgecolor='k'
        )

        # Test como puntos con error
        subset_test = df[(df["class"] == clase) & (df["set"] == "Test")]
        counts_test, _ = np.histogram(
            subset_test["score"], bins=bin_edges, density=True
        )
        errors = np.sqrt(counts_test / len(subset_test))  # Aproximación poissoniana
        plt.errorbar(
            bin_centers, counts_test, yerr=errors,
            fmt='o', label=f"{clase} (Test)",
            color=palette[clase], capsize=2
        )

    # KS test
    ks_s, p_s = ks_2samp(
        df_train[df_train["label"] == 1]["score"],
        df_test[df_test["label"] == 1]["score"]
    )
    ks_b, p_b = ks_2samp(
        df_train[df_train["label"] == 0]["score"],
        df_test[df_test["label"] == 0]["score"]
    )

    # Plot final
    plt.title("Comparación de Score (Train: barras, Test: puntos) por clase")
    plt.xlabel("Score del modelo (probabilidad de fraude)")
    plt.ylabel("Densidad normalizada")
    plt.legend(title="Distribución")
    plt.text(
        0.5, 0.95,
        f"KS p-value Señal: {p_s:.3f} | Background: {p_b:.3f}",
        ha="center", va="top", transform=plt.gca().transAxes,
        fontsize=10, bbox=dict(facecolor="white", alpha=0.7)
    )
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, threshold=None, save_path=None, target="Class", top_n=10):
    """
    Visualiza la matriz de correlación entre variables numéricas y muestra las top correlaciones con la variable target.

    Parámetros:
    - df: DataFrame con los datos
    - threshold: si se define, se mostrarán solo correlaciones con |r| > threshold
    - save_path: si se define, guarda el gráfico como imagen
    - target: nombre de la columna target (por defecto: 'Class')
    - top_n: número de variables más correlacionadas con el target a mostrar
    """

    # Asegurar que target esté en el dataframe
    if target not in df.columns:
        raise ValueError(f"La columna target '{target}' no se encuentra en el DataFrame.")

    # Filtrar solo variables numéricas
    numeric_df = df.select_dtypes(include=[np.number])

    # Calcular matriz de correlación
    corr = numeric_df.corr()

    # Mostrar correlaciones con el target
    if target in corr.columns:
        top_corr = corr[target].drop(target).abs().sort_values(ascending=False).head(top_n)
        print(f"\n🔎 Top {top_n} variables más correlacionadas con '{target}':\n")
        print(top_corr.to_frame("Correlación").round(3))
    else:
        print(f"[!] Advertencia: '{target}' no está en las columnas numéricas para correlación.")

    # Aplicar umbral (opcional)
    if threshold:
        corr = corr.where(np.abs(corr) > threshold)

    # Crear máscara para ocultar diagonal superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Plot
    sns.set(style="white")
    f, ax = plt.subplots(figsize=(10, 8))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8}
    )

    plt.title("Matriz de correlación entre variables numéricas")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()