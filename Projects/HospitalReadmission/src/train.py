import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight


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
    También se incluye Dropout para combatir el sobreajuste.
    """
    model = Sequential()
    for i, units in enumerate(config["model"]["hidden_layers"]):
        if i == 0:
            model.add(Dense(units,
                            activation=config["model"]["activation"],
                            input_shape=(input_dim,),
                            kernel_regularizer=l2(0.001)))
        else:
            model.add(Dense(units,
                            activation=config["model"]["activation"],
                            kernel_regularizer=l2(0.001)))
        
        model.add(Dropout(0.3))  # 👈 Dropout de 30% después de cada capa oculta

    model.add(Dense(1, activation=config["model"]["output_activation"]))

    model.compile(optimizer=Adam(learning_rate=config["learning_rate"]),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Entrenamiento del modelo
def train_model(model, X_train, y_train, config):
    """
    Entrena la red neuronal usando los datos de entrenamiento.
    Aplica EarlyStopping para detener entrenamiento si no mejora la pérdida de validación.
    Aplica pesos de clase para manejar desbalance.
    """
    # Calcular pesos de clase automáticamente
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(y_train),
                                         y=y_train)
    class_weights = {i: w for i, w in enumerate(class_weights)}
    print("Pesos de clase:", class_weights)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    return model.fit(
        X_train, y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_split=0.2,
        callbacks=[early_stop],
        class_weight=class_weights, 
        verbose=1
    )