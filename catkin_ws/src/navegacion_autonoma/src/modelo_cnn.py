from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, ReLU

def build_cnn_model(input_shape=(64, 64, 3)):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), strides=2, padding='same', input_shape=input_shape))
    model.add(ReLU())
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), strides=2, padding='same'))
    model.add(ReLU())
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), strides=2, padding='same'))
    model.add(ReLU())
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256))
    model.add(ReLU())
    model.add(Dense(64))
    model.add(ReLU())
    model.add(Dense(2))  # Salida: [v, w]
    return model
