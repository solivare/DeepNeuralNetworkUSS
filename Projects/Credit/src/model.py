"""
build_model v2
- soporta arquitecturas arbitrarias definidas en config.yaml
- añade Dropout y regularización L2 parametrizable
- mantiene compatibilidad con fit() en train.py
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


# CHANGE v2: función ahora usa claves nuevas del YAML
def build_model(cfg, input_dim):
    """
    Construye una red densa según parámetros de `cfg`.
    - Capas ocultas: cfg["model"]["layers"]
    - Activación     cfg["model"]["activation"]
    - Dropout        cfg["model"]["dropout"]
    - L2             cfg["model"]["l2"]
    """
    model = Sequential()

    for i, units in enumerate(cfg["model"]["layers"]):
        model.add(
            Dense(
                units,
                activation=cfg["model"]["activation"],
                input_shape=(input_dim,) if i == 0 else None,
                kernel_regularizer=l2(cfg["model"]["l2"])      # CHANGE v2
            )
        )
        model.add(Dropout(cfg["model"]["dropout"]))             # CHANGE v2

    # Capa de salida binaria
    model.add(Dense(1, activation="sigmoid"))

    # Optimizador con LR configurable
    opt = Adam(learning_rate=cfg["learning_rate"])
    model.compile(optimizer=opt,
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model