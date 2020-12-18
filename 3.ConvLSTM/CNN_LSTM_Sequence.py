import img_util
from pathlib import Path
import numpy as np
import keras.models
from keras.layers import InputLayer, LSTM, TimeDistributed, Dense, Dropout, Conv2D, MaxPooling2D, Flatten


def generate_data(real_sequences, fake_sequences):
    X = [*real_sequences, *fake_sequences]
    X = np.asarray(X)
    Y_fake = np.zeros(len(real_sequences))
    Y_real = np.ones(len(fake_sequences))
    Y = np.concatenate([Y_real, Y_fake])
    return X, Y


def generate_model():
    model = keras.models.Sequential()
    model.add(InputLayer(input_shape=(10, 40, 40, 3)))
    model.add(
        TimeDistributed(
            Conv2D(32, (3, 3), activation='relu'),
            input_shape=(10, 40, 40, 3)
        )
    )
    
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    # extract features and dropout
    model.add(Flatten())
    # classifier with sigmoid activation for multilabel
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss="binary_crossentropy", optimizer="adadelta")
    model.summary()

    return model

def train_model(model, X, Y):
    model.fit(
        x=X, y=Y,
        epochs=100,
        validation_split=0.1,
    )

def main():
    real_path = Path("../faces_resized/real")
    fake_path = Path("../faces_resized/fake")

    real_sequences = img_util.load_sequences(real_path)
    fake_sequences = img_util.load_sequences(fake_path)

    print("El tamaño de la sequencia real es de {} elementos y de falsos es de {}, el combinado es {}.".format(
        len(real_sequences), len(fake_sequences), len(real_sequences) + len(fake_sequences)))

    X, Y = generate_data(real_sequences, fake_sequences)

    print("El tañamo de X es {} y de Y {}.".format(len(X), len(Y)))
    print("La forma de X es {}.", format(X.shape))

    model = generate_model()

    train_model(model,X,Y)

if __name__ == "__main__":
    main()
