import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight  
from tensorflow.keras.callbacks import EarlyStopping 
from model import build_model



def load_config(path="../config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(path="../data/credit_clean.csv"):
    df = pd.read_csv(path)
    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]
    return X, y

def preprocess(X, y, config):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(
        X_scaled, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y
    )

def train_model(model, X_train, y_train, config):
    # Calcular pesos de clase
    cw = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    cw_dict = {0: cw[0], 1: cw[1]}
    
    # Callback de early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    return model.fit(
        X_train, y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_split=0.2,
        callbacks=[early_stop],
        class_weight=cw_dict,
        verbose=1
    )