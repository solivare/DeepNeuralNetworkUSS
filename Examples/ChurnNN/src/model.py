
from tensorflow import keras
from tensorflow.keras import layers

def build_model(config):
    """
    Construye un modelo secuencial Keras utilizando los parámetros especificados en config.yaml.
    
    Arquitectura:
    - Capa de entrada: 4 características (compras, llamadas, etc.)
    - Capas ocultas: definidas en 'hidden_units' con activación especificada
    - Capa de salida: 1 neurona con activación sigmoide (clasificación binaria)

    Devuelve un modelo compilado listo para entrenar.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(4,)))  # 4 features de entrada

    # Capas ocultas definidas por configuración
    for units in config['model']['hidden_units']:
        model.add(layers.Dense(units, activation=config['model']['activation']))
        model.add(layers.Dropout(config['model']['dropout_rate']))


    # Capa de salida para clasificación binaria
    model.add(layers.Dense(1, activation=config['model']['output_activation']))

    # Compilar modelo
    model.compile(
        optimizer=config['training']['optimizer'],
        loss=config['training']['loss'],
        metrics=['accuracy']
    )

    return model
