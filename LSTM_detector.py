import json
from pathlib import Path

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from keras.layers import Dense, Dropout, LSTM, TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.backend import tensorflow_backend
from sklearn.model_selection import train_test_split
import numpy as np

def check_gpu():
    tensorflow_backend._get_available_gpus()

def load_landmarks(path):
    files = [x for x in path.iterdir() if x.is_file() ]
    data = []
    
    for file in files:
        video_landmarks = []

        with open(file) as f:
            file_data = json.load(f)

        if len(file_data.keys()) != 251:
            print(str(file) + ' tiene una cantidad de frames distinta a la requerida')

        for key in sorted(file_data.keys(), key=lambda x: int(Path(x).stem.replace('frame',''))):
            frame_lm = np.asarray(file_data[key])

            for lm in frame_lm:
                lm[0] = np.interp(lm[0],[0,1920],[-1,1])
                lm[1] = np.interp(lm[1],[0,1080],[-1,1])

            video_landmarks.append(frame_lm)
        data.append(np.asarray(video_landmarks))
    return np.asarray(data)
        
def load_data(data_path):
    fake_data_path = data_path / 'fakes'
    real_data_path = data_path / 'reales'

    if not fake_data_path.exists() or not real_data_path.exists():
        print('No se encuentran las carpetas de landmarks')
    else:
        fake_data = load_landmarks(fake_data_path)
        real_data = load_landmarks(real_data_path)
        return fake_data, real_data

def data_treatment(data_path):
    fake_data, real_data = load_data(data_path)

    fake_data_labels = np.zeros(fake_data.shape[0])
    real_data_labels = np.ones(real_data.shape[0])

    data = np.concatenate((fake_data,real_data), axis = 0)
    labels = np.concatenate((fake_data_labels,real_data_labels), axis = 0)
    data = data.reshape(data.shape[0],data.shape[1],data.shape[2]*data.shape[3])
    labels = labels.reshape(labels.shape[0],1)
    return data,labels
    


def create_model():
    # La capa LSTM espera una entrada 3D tal que [samples, timesteps, features]
    print ('Creando el modelo.')
    model = Sequential()
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    print ('Compilando el modelo.')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def train_model(model, data, labels, epochs):
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2)
    return model.fit(data_train, labels_train, epochs=1000, batch_size=100, validation_data=(data_test, labels_test), verbose=1)

def save_model(model, name):
    with open(name + ".json", "w") as json_file:
        json_file.write(model.to_json())

def save_weights(model, name):
    model.save_weights(name + ".h5")
    print("Saved model to disk")

def main():
    model = create_model()
    save_model(model, 'latest')

    data,labels = data_treatment(Path('./landmarks'))
    
    train_model(model, data, labels,10000)
    save_weights(model,'latest')

if __name__ == "__main__":
    main()