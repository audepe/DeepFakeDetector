import img_util
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
                filters=10, kernel_size=(4, 4), padding="same", return_sequences=True
            ),
            layers.Flatten(),
            layers.Dense(
                 1,activation="sigmoid"
            ),
        ]
    )
    seq.compile(loss="binary_crossentropy",metrics="accuracy", optimizer="adadelta")
    seq.summary()
    return seq
def train_model(model, X, Y):
    model.fit(
        x=X, y=Y,
        batch_size=1,
        epochs=100,
        validation_split=0.1,
        verbose=1        
    )
real_path = Path("../faces_resized/real")
fake_path = Path("../faces_resized/fake")
real_sequences = img_util.load_sequences(real_path)
fake_sequences = img_util.load_sequences(fake_path)
print("El tamaño de la sequencia real es de {} elementos y de falsos es de {}, el combinado es {}.".format(
    len(real_sequences), len(fake_sequences), len(real_sequences) + len(fake_sequences)))
try:
    X, Y = generate_data(real_sequences, fake_sequences)
except Exception as ex:
    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    print (message)
print("El tañamo de X es {} y de Y {}.".format(len(X), len(Y)))
print("La forma de X es {}.", format(X.shape))
model = create_model()
train_model(model, X, Y)