import pandas as pd
import numpy as np
import os

# Rutas de entrada y salida
INPUT_PATH = "data/diabetic_data.csv"
OUTPUT_PATH = "data/hospital_readmission_clean.csv"

# Carga inicial de datos
def load_data(path):
    """
    Carga el archivo CSV original entregado por UCI.
    """
    print(f"Cargando datos desde {path}")
    return pd.read_csv(path)

# Limpieza inicial del dataset
def clean_data(df):
    """
    Reemplaza valores faltantes y elimina columnas irrelevantes o poco informativas.
    """
    print("Reemplazando '?' por NaN...")
    df.replace('?', np.nan, inplace=True)

    print("Valores NaN por columna:")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    print("\nEliminando columnas irrelevantes o con muchos NaNs...")
    columns_to_drop = ['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    print("Eliminando filas con NaN en columnas críticas (race, gender, age)...")
    df.dropna(subset=['race', 'gender', 'age'], inplace=True)

    return df

# Transformación de la variable objetivo
def transform_target(df):
    """
    Convierte la variable `readmitted` en binaria:
    1 si fue readmitido en menos de 30 días, 0 en cualquier otro caso.
    """
    print("Convirtiendo variable objetivo...")
    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    return df

# Eliminación de columnas sin varianza
def drop_low_variance(df):
    """
    Elimina columnas que solo contienen un valor (no aportan a la predicción).
    """
    print("Eliminando columnas con un solo valor único...")
    nunique = df.nunique()
    low_var_cols = nunique[nunique <= 1].index.tolist()
    df.drop(columns=low_var_cols, inplace=True)
    return df

# Codificación one-hot para variables categóricas
def encode_categoricals(df):
    """
    Aplica one-hot encoding a las variables categóricas (menos la variable objetivo).
    """
    print("Codificando variables categóricas...")
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'readmitted']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

# Guardado final del dataset limpio
def save_data(df, path):
    """
    Guarda el dataset limpio y muestra información útil.
    """
    print(f"Guardando dataset limpio en {path}")
    df.to_csv(path, index=False)
    print("✅ Preprocesamiento finalizado.")
    print(f"📦 Registros finales: {df.shape[0]}  |  Variables: {df.shape[1]}")
    print(f"🔍 Distribución del target:\n{df['readmitted'].value_counts(normalize=True)}")

# Ejecución principal
def main():
    """
    Flujo completo de preprocesamiento.
    """
    df = load_data(INPUT_PATH)
    df = clean_data(df)
    df = transform_target(df)
    df = drop_low_variance(df)
    df = encode_categoricals(df)
    save_data(df, OUTPUT_PATH)

# Punto de entrada si se ejecuta como script
if __name__ == "__main__":
    main()