import json
from pathlib import Path
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import talos
from talos.utils import lr_normalizer
from keras.layers import Dense, Dropout, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.optimizers import Adam, Nadam, RMSprop
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
    


def create_model(params):
    # La capa LSTM espera una entrada 3D tal que [samples, timesteps, features]
    print ('Creando el modelo.')
    model = Sequential()
    model.add(LSTM(params['first_layer'], activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True))
    model.add(Dropout(params['dropout']))
    model.add(Dense(1, activation=params['last_activation']))

    print ('Compilando el modelo.')
    model.compile(loss=params['losses'],
                  optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  metrics=['acc'])
    return model

def train_model(model, x_train, y_train, x_val, y_val, params):
    #data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2)
    return model.fit(x_train, y_train, 
                    epochs=params['epochs'], 
                    validation_data=(x_val, y_val), 
                    verbose=2, shuffle=False)

def compile_and_train(x_train, y_train, x_val, y_val, params):
    model = create_model(params)
    history = train_model(model, x_train, y_train, x_val, y_val, params)
    
    return history,model

def main():
    p = {
     'first_layer': [100, 150, 200],
     'epochs': [100],
     'dropout': (0.2, 0.5, 2),
     'lr': (0.1, 10, 5),
     'optimizer': [Adam, Nadam],
     'losses': ['binary_crossentropy'],
     'last_activation': ['sigmoid']}

    data,labels = data_treatment(Path('./landmarks'))

    t = talos.Scan(x=data,
        y=labels,
        model=compile_and_train,
        params=p,
        experiment_name='talos_lstm')

    analyze_object = talos.Analyze(t)

    print('Los mejores resultados han sido')
    print(analyze_object.high('val_acc'))


if __name__ == "__main__":
    main()