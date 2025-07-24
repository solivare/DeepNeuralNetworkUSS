from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def build_model(config, input_dim):
    """
    Construye una red neuronal basada en los parámetros definidos en config.yaml
    """
    model = Sequential()
    for i, units in enumerate(config["model"]["hidden_layers"]):
        if i == 0:
            model.add(Dense(units, activation=config["model"]["activation"],
                            input_shape=(input_dim,),
                            kernel_regularizer=l2(0.001)))
        else:
            model.add(Dense(units, activation=config["model"]["activation"],
                            kernel_regularizer=l2(0.001)))

    model.add(Dense(1, activation=config["model"]["output_activation"]))

    model.compile(optimizer=Adam(learning_rate=config["learning_rate"]),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
