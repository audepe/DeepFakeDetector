import LSTM_detector as ld
import numpy as np
import json
from pathlib import Path

def test_load_landmarks():
    data = ld.load_landmarks(Path('./tlm_reales'), 3)
    if data is not None:
        print('Dimensiones del directorio cargado')
        print(data.shape)
        print('Los landmarks cargan correctamente.')

def test_model_compile():
    model = ld.create_model()
    if model is not None:
        print('El modelo compila correctamente.')
        return model

def test_load_data():
    fake_data, real_data = ld.load_data(Path('./landmarks'), 3)
    if real_data is not None and fake_data is not None:
        print('Dimensiones del directorio real')
        print(real_data.shape)
        print('Dimensiones del directorio fake')
        print(fake_data.shape)
        print('Los datos cargan correctamente.')

def test_data_treatment():
    data,labels = ld.data_treatment(Path('./landmarks'), 3)
    if data is not None and labels is not None:
        if data.shape[0] == labels.shape[0]:
            print('Dimension de los datos')
            print(data.shape)
            print('Dimension de las etiquetas')
            print(labels.shape)
            print('El tratamiento de datos es correcto')

def test_train():
    model = ld.create_model()
    data,labels = ld.data_treatment(Path('./tlm_reales'), 3)
    if ld.train_model(model, data, labels,1) is not None:
        print('El modelo entrena')

def print_model_summary():
    summary = test_model_compile().summary()
    print(summary)

def test_multipredict():
    data = ld.load_landmarks(Path('./split_test'), 3)
    data = data.reshape(data.shape[0],data.shape[1],data.shape[2]*data.shape[3])
    predictions = ld.load_and_predict('model-and-weights.h5', data, 3)
    print('Shape de las predicciones.')
    print(predictions.shape)

def generate_predictions(parts):
    data,labels = ld.data_treatment(Path('./landmarks'), parts)
    predictions = ld.load_and_predict('model-and-weights.h5', data, parts)

    predictions_list = []

    for pred in predictions:
        predictions_list.append(pred.tolist())

    labels = labels.tolist()

    pred_dict = {
        "preds" : predictions_list,
        "labels" : labels
    }

    with open('preds-out-dict.json', "w") as write_file:
        json.dump(pred_dict, write_file)

def generate_outcomes():
    predictions,labels = ld.load_predictions(Path('./preds-out-dict.json'))
    thresholds = [0.5, 0.7, 0.9]

    scored_preds = []

    for threshold in thresholds:
        scored_preds.append(ld.apply_threshold(threshold, predictions))
    
    preds_by_threshold = {}

    for idx,pred in enumerate(scored_preds):  
        group_of_preds = []

        for i in range(4):                
            group_of_preds.append(ld.get_precision_recall(pred, labels, i))

        preds_by_threshold[str(thresholds[idx])] = group_of_preds
    
    return preds_by_threshold
        

def main():
    # test_load_landmarks()
    # test_model_compile()
    # test_load_data()
    # test_data_treatment()
    # print_model_summary()
    # test_train()
    # test_multipredict()
    # generate_predictions(3)
    generate_outcomes()

if __name__ == "__main__":
    main()