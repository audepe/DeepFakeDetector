import json
from pathlib import Path
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from keras.layers import Dense, Dropout, LSTM, TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.backend import tensorflow_backend
from sklearn.model_selection import train_test_split
import numpy as np

def check_gpu():
    tensorflow_backend._get_available_gpus()

def load_landmarks(path, video_parts):
    files = [x for x in path.iterdir() if x.is_file() ]
    data = []

    for file in files:
        video_landmarks = []
        with open(file) as f:
            file_data = json.load(f)
        if len(file_data.keys()) < 251:
            print(str(file) + ' tiene una cantidad de frames menor a la requerida')
            continue

        for key in sorted(file_data.keys(), key=lambda x: int(Path(x).stem.replace('frame','')))[:nearest_partition(len(file_data.keys()), video_parts)]:
            frame_lm = np.asarray(file_data[key])
            for lm in frame_lm:
                lm[0] = np.interp(lm[0],[0,1920],[-1,1])
                lm[1] = np.interp(lm[1],[0,1080],[-1,1])
            past_frame = frame_lm[0]
            for lm in frame_lm:
                aux = past_frame - lm
                past_frame = lm
                lm = aux
            
            
            video_landmarks.append(frame_lm)

        video_landmarks = np.asarray(video_landmarks)
        for part in split_array(video_landmarks, video_parts):
            data.append(np.asarray(part))
            
    return np.asarray(data)
        
def load_data(data_path, video_parts):
    print('Cargando datos.')
    fake_data_path = data_path / 'fakes'
    real_data_path = data_path / 'reales'
    if not fake_data_path.exists() or not real_data_path.exists():
        print('No se encuentran las carpetas de landmarks')
    else:
        fake_data = load_landmarks(fake_data_path, video_parts)
        real_data = load_landmarks(real_data_path, video_parts)
        print('Datos cargados.')
        return fake_data, real_data

def data_treatment(data_path, video_parts):
    print('Procesando datos.')
    fake_data, real_data = load_data(data_path, video_parts)
    fake_data_labels = np.zeros(fake_data.shape[0])
    real_data_labels = np.ones(real_data.shape[0])
    data = np.concatenate((fake_data,real_data), axis = 0)
    labels = np.concatenate((fake_data_labels,real_data_labels), axis = 0)
    data = data.reshape(data.shape[0],data.shape[1],data.shape[2]*data.shape[3])
    labels = labels.reshape(labels.shape[0],1)
    print('Datos procesados.')
    print('Shape procesado:')
    print(data.shape)
    return data,labels
    
def create_model():
    # La capa LSTM espera una entrada 3D tal que [samples, timesteps, features]
    print ('Creando el modelo.')
    model = Sequential()
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
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
    return model.fit(data_train, labels_train, epochs=epochs, validation_data=(data_test, labels_test), verbose=1)

def save_model(model, name):
    with open(name + ".json", "w") as json_file:
        json_file.write(model.to_json())
        print("Modelo guardado.")

def save_weights(model, name):
    model.save_weights(name + ".h5")
    print("Pesos guardados.")

def save_model_and_weights(model, name):
    model.save(str(name) + '.h5')
    print('Modelo y pesos guardados.')


def nearest_partition(total, parts):
    for i in reversed(range(total)):
        if (i+1) % parts == 0:
            return i+1

def split_array(arr, parts):
    minimun_frames = nearest_partition(arr.shape[0],parts)
    arr = arr[:minimun_frames]

    splits = []
    step = int(minimun_frames / parts)
    for i in range(parts):
        splits.append( arr[step*i:step*(i+1)] )

    return splits 
        

def multi_predict(video, parts, model):
    predictions = []

    for part in range(parts):
        print('Shape de las partes')
        print(video[part].shape)
        predict = model.predict(video[part].reshape(1,video[part].shape[0],video[part].shape[1]))
        print('Shape de una predicción.')
        print(predict.shape)
        predictions.append(predict)

    predictions = np.asarray(predictions)

    return predictions.reshape(predictions.shape[0])

def load_and_predict(path, video, parts):
    model = load_model(path)
    print('Shape de los datos.')
    print(video.shape)
    print('Partes a dividir.')
    print(parts)
    predictions = multi_predict(video,parts,model)
    return predictions

def main():
    model = create_model()
    data,labels = data_treatment(Path('./landmarks'), 3)
    
    train_model(model, data, labels, 100)
    save_model_and_weights(model, 'model-and-weights')
if __name__ == "__main__":
    main()