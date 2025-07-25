import pandas as pd
import numpy as np
import os

# Paths de entrada y salida
INPUT_PATH = "../data/cs-training.csv"
OUTPUT_PATH = "../data/credit_clean.csv"

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
    """
    Genera nuevas variables que combinan información existente para
    mejorar el poder predictivo del modelo.  En concreto:

    - ``TotalPastDue``: suma del número de retrasos de 30–59, 60–89 y >=90 días.
    - ``MonthlyDebt``: estimación del pago de deuda mensual calculado como
      ``DebtRatio * MonthlyIncome``.
    - ``UtilizationPerLine``: ratio de utilización de líneas revolving por
      cada línea de crédito abierta.
    - ``RealEstateLoanRatio``: proporción de préstamos inmobiliarios
      respecto al número total de líneas de crédito abiertas.

    Cualquier valor infinito o NaN generado por divisiones se sustituye
    por 0.
    """
    df = df.copy()
    # Total de retrasos acumulados
    df['TotalPastDue'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse']
        + df['NumberOfTime60-89DaysPastDueNotWorse']
        + df['NumberOfTimes90DaysLate']
    )
    # Estimación de deuda mensual
    df['MonthlyDebt'] = df['DebtRatio'] * df['MonthlyIncome']
    # Utilización promedio por línea de crédito (evitar dividir por cero)
    df['UtilizationPerLine'] = df['RevolvingUtilizationOfUnsecuredLines'] / (
        df['NumberOfOpenCreditLinesAndLoans'] + 1e-6
    )
    # Proporción de préstamos inmobiliarios sobre líneas abiertas
    df['RealEstateLoanRatio'] = df['NumberRealEstateLoansOrLines'] / (
        df['NumberOfOpenCreditLinesAndLoans'] + 1e-6
    )
    # Sustituir inf o NaN por 0
    for col in ['UtilizationPerLine', 'RealEstateLoanRatio']:
        df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
    return df

def remove_outliers(df: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    """
    Recorta cada variable numérica (excepto el target) a los cuantiles
    especificados.  Esto atenúa la influencia de valores extremos sin
    eliminar registros completos.

    :param df: DataFrame de entrada.
    :param lower_q: cuantil inferior (por defecto 0.01).
    :param upper_q: cuantil superior (por defecto 0.99).
    :returns: DataFrame con las columnas numéricas recortadas.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'SeriousDlqin2yrs':
            lower = df[col].quantile(lower_q)
            upper = df[col].quantile(upper_q)
            df[col] = df[col].clip(lower, upper)
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
    df = remove_outliers(df)
    df = feature_engineering(df)
    df = balance_dataset(df)
    save_data(df, OUTPUT_PATH)

if __name__ == "__main__":
    main()
