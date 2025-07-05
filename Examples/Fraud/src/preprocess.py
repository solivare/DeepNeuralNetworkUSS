import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml

def load_config(config_filename="config.yaml"):
    base_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_path, ".."))
    config_path = os.path.join(root_dir, config_filename)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["base_dir"] = root_dir  # Esto permite rutas absolutas
    return config

def load_and_preprocess_data(config):
    filepath = os.path.join(config["base_dir"], config["paths"]["original"])
    output_dir = os.path.join(config["base_dir"], config["paths"]["processed"])
    sample_size = config["preprocessing"].get("sample_size", None)
    subsample = config["preprocessing"].get("subsample", False)
    random_state = config["preprocessing"].get("random_state", 42)

    # Cargar datos
    df = pd.read_csv(filepath)
    print(f"Dataset original: {df.shape[0]} filas, {df.shape[1]} columnas")

    # Filtrado opcional (reducir tamaño del dataset completo)
    if sample_size is not None and sample_size < len(df):
        df = df.groupby('Class', group_keys=False).apply(
            lambda x: x.sample(int(sample_size * len(x) / len(df)), random_state=random_state)
        ).reset_index(drop=True)
        print(f"Dataset reducido a: {df.shape[0]} filas (manteniendo proporción de fraude)")

    # Guardar copia completa para entrenamiento con class_weight
    df_train_full, df_temp = train_test_split(df, test_size=0.3, stratify=df["Class"], random_state=random_state)
    df_val_full, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp["Class"], random_state=random_state)

    os.makedirs(output_dir, exist_ok=True)
    df_train_full.to_csv(os.path.join(output_dir, "train_full.csv"), index=False)
    df_val_full.to_csv(f"{output_dir}/val.csv", index=False)
    df_test.to_csv(f"{output_dir}/test.csv", index=False)
    print("✅ Archivos completos guardados (train_full, val, test)")

    # Si subsample está activado, crear dataset balanceado
    if subsample:
        df_fraud = df[df["Class"] == 1]
        df_nonfraud = df[df["Class"] == 0].sample(len(df_fraud), random_state=random_state)
        df_balanced = pd.concat([df_fraud, df_nonfraud]).sample(frac=1, random_state=random_state)

        df_train, df_temp = train_test_split(df_balanced, test_size=0.3, stratify=df_balanced["Class"], random_state=random_state)
        df_val, df_test_bal = train_test_split(df_temp, test_size=0.5, stratify=df_temp["Class"], random_state=random_state)

        df_train.to_csv(f"{output_dir}/train.csv", index=False)
        df_val.to_csv(f"{output_dir}/val.csv", index=False)
        df_test_bal.to_csv(f"{output_dir}/test_balanced.csv", index=False)
        print("✅ Archivos balanceados guardados (train, val, test_balanced)")

if __name__ == "__main__":
    config = load_config()
    load_and_preprocess_data(config)