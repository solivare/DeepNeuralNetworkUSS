import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Cargar configuración desde archivo YAML
def load_config(path="../config.yaml"):
    """
    Carga los parámetros de configuración definidos en un archivo YAML.
    Esto incluye tamaño del batch, número de épocas, estructura del modelo, etc.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Cargar datos preprocesados
def load_data(path="../data/hospital_readmission_clean.csv"):
    """
    Carga el dataset procesado y separa características (X) y variable objetivo (y).
    """
    df = pd.read_csv(path)
    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]
    return X, y

# Normalización y división del dataset
def preprocess(X, y, config):
    """
    Escala las características numéricas usando StandardScaler
    y separa en conjuntos de entrenamiento y prueba (stratified).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(
        X_scaled, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y  # asegura que la proporción de clases se mantenga
    )

# Construcción de la red neuronal
def build_model(config, input_dim):
    """
    Construye una red neuronal densa con Keras.
    Se permite definir el número de capas ocultas, neuronas y activación desde YAML.
    Se aplica regularización L2 para evitar sobreajuste.
    """
    model = Sequential()
    for i, units in enumerate(config["model"]["hidden_layers"]):
        if i == 0:
            # Primera capa: requiere definir input_shape
            model.add(Dense(units, activation=config["model"]["activation"],
                            input_shape=(input_dim,),
                            kernel_regularizer=l2(0.001)))
        else:
            model.add(Dense(units, activation=config["model"]["activation"],
                            kernel_regularizer=l2(0.001)))

    # Capa de salida con activación sigmoide para clasificación binaria
    model.add(Dense(1, activation=config["model"]["output_activation"]))
    
    # Compilación del modelo con optimizador Adam y pérdida binaria
    model.compile(optimizer=Adam(learning_rate=config["learning_rate"]),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Entrenamiento del modelo
def train_model(model, X_train, y_train, config):
    """
    Entrena la red neuronal usando los datos de entrenamiento.
    Se reserva automáticamente un 20% de los datos para validación.
    """
    return model.fit(
        X_train, y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_split=0.2  # importante para monitorear sobreajuste
    )