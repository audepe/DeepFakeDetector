import img_util
import json
from custom_data_gen import DataGenerator
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping


def generate_data(real_sequences, fake_sequences):
    X = [*real_sequences, *fake_sequences]
    X = np.asarray(X)
    Y_fake = np.zeros(len(real_sequences))
    Y_real = np.ones(len(fake_sequences))
    Y = np.concatenate([Y_real, Y_fake])
    return X, Y


def create_model():
    seq = Sequential(
        [
            layers.InputLayer(
                input_shape=(251, 64, 64, 3)
            ),  # Variable-length sequence of 40x40x1 frames
            layers.ConvLSTM2D(
                filters=5, kernel_size=(16, 16), padding="same", return_sequences=True
            ),
            layers.Flatten(),
            layers.Dense(
                1, activation="sigmoid"
            ),
        ]
    )
    seq.compile(loss="binary_crossentropy",
                metrics="accuracy", optimizer="adadelta")
    seq.summary()
    return seq


def train_model(model, data_gen_train, data_gen_test):
    earlystop = EarlyStopping(patience=7)
    callbacks = [earlystop]
    history = model.fit_generator(
        epochs=20,
        generator=data_gen_train,
        validation_data=data_gen_test,
        callbacks=callbacks,
        verbose=1
    )
    return history


def main():
    with open("sequences.json", "r") as read_file:
        sequences = json.load(read_file)

    split_index = int(np.floor(len(sequences) * 0.8))

    model = create_model()

    batch_size = 5
    data_gen_train = DataGenerator(batch_size, dict(
        list(sequences.items())[:split_index]))
    data_gen_test = DataGenerator(batch_size, dict(
        list(sequences.items())[split_index:]))
    history = train_model(model, data_gen_train, data_gen_test)
    model.save("model_and_weights.h5")
    with open("history.json", "w") as write_file:
        json.dump(history.history, write_file)


if __name__ == "__main__":
    main()
