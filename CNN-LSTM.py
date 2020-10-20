from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam
import argparse
from pathlib import Path

# # Definición de parámetros
# parser = argparse.ArgumentParser()
# parser.add_argument('--frames', type=int, required=True, help="Número de frames a usar.")
# args = parser.parse_args()

def directory_tree_load(path):
    #Carga de imágenes
    data_path = Path(path)

    videos = []
    for subdir in [f for f in data_path.iterdir() if f.is_dir()]:
        video = []
        for file in [f for f in subdir.iterdir() if f.is_file()]:
            video.append(Image.open(str(file)))
        videos.append(video)
    
    return videos


def def_and_compile(frames, channels, rows, colmns):
    video = Input(shape=(frames,
                        channels,
                        rows,
                        columns))
    cnn_base = VGG16(input_shape=(channels,
                                rows,
                                columns),
                    weights="imagenet",
                    include_top=False)
    cnn_out = GlobalAveragePooling2D()(cnn_base.output)
    cnn = Model(input=cnn_base.input, output=cnn_out)
    cnn.trainable = False
    encoded_frames = TimeDistributed(cnn)(video)
    encoded_sequence = LSTM(256)(encoded_frames)
    hidden_layer = Dense(output_dim=1024, activation="relu")(encoded_sequence)
    outputs = Dense(output_dim=classes, activation="softmax")(hidden_layer)
    model = Model([video], outputs)
    optimizer = Nadam(lr=0.002,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-08,
                    schedule_decay=0.004)
    model.compile(loss="categorical_crossentropy",
                optimizer=optimizer,
                metrics=["categorical_accuracy"])

def main():
    # len(directory_tree_load('/home/dadepe/Imágenes/frames/fake'))
    datagen = ImageDataGenerator()
    train = datagen.flow_from_directory('/home/dadepe/Imágenes/frames/', class_mode='binary')

if __name__ == "__main__":
    main()