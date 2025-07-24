import pandas as pd
import numpy as np
import os

# Paths de entrada y salida
INPUT_PATH = "data\cs-training.csv"
OUTPUT_PATH = "data\credit_clean.csv"

def load_data(path):
    print(f"📥 Cargando datos desde {path}")
    return pd.read_csv(path, index_col=0)

def clean_data(df):
    print("🧼 Eliminando valores extremos y codificando missing...")

    # Reemplazar valores -1 o 0 por NaN donde corresponda
    df['MonthlyIncome'].replace(0, np.nan, inplace=True)
    df['MonthlyIncome'].replace(-1, np.nan, inplace=True)
    df['NumberOfDependents'].replace(-1, np.nan, inplace=True)

    # Eliminar duplicados si existen
    df.drop_duplicates(inplace=True)

    return df

def impute_missing(df):
    print("🔄 Imputando valores faltantes...")
    df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
    df['NumberOfDependents'].fillna(df['NumberOfDependents'].mode()[0], inplace=True)
    return df

def feature_engineering(df):
    print("🛠️ Ingeniería de características...")
    # Aquí podrían agregarse variables nuevas o transformaciones
    return df

def balance_dataset(df, target_col="SeriousDlqin2yrs", max_negatives=20000, positives_multiplier=2):
    print("⚖️ Balanceando dataset...")
    positives = df[df[target_col] == 1]
    negatives = df[df[target_col] == 0].sample(
        n=min(max_negatives, len(df[df[target_col] == 0])), random_state=42
    )

    sampled_df = pd.concat([positives] * positives_multiplier + [negatives], axis=0)
    sampled_df = sampled_df.sample(frac=1, random_state=42)  # Mezclar
    return sampled_df

def save_data(df, path):
    print(f"💾 Guardando dataset limpio en {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Registros: {df.shape[0]} | Variables: {df.shape[1]}")
    print("📊 Distribución del target:")
    print(df["SeriousDlqin2yrs"].value_counts(normalize=True))

def main():
    df = load_data(INPUT_PATH)
    df = clean_data(df)
    df = impute_missing(df)
    df = feature_engineering(df)
    df = balance_dataset(df)
    save_data(df, OUTPUT_PATH)

if __name__ == "__main__":
    main()
