import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import build_model  # Importar desde model.py
from sklearn.impute import SimpleImputer

def load_config(path="../config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(path="../data/credit_clean.csv"):
    df = pd.read_csv(path)
    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]
    return X, y

def preprocess(X, y, config):
    # Imputar valores faltantes con la media
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Dividir datos
    return train_test_split(
        X_scaled, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y
    )
    
def train_model(model, X_train, y_train, config):
    return model.fit(
        X_train, y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_split=0.2,
        verbose=1
    )