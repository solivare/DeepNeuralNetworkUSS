from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(config):
    """
    Construye una red CNN a partir de la configuración
    """
    model = Sequential()

    # Capa convolucional 1
    model.add(Conv2D(32, (3, 3), activation='relu',
                     input_shape=(config["image"]["height"], config["image"]["width"], 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Capa convolucional 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Capa convolucional 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Capa densa intermedia
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Capa de salida
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate = config["training"].get("learning_rate", 0.001))

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model