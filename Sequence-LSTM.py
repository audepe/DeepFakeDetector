from pathlib import Path
from keras.preprocessing.image import load_img
import numpy as np
import cv2

def load_sequences(path):
    subdirs = [x for x in path.iterdir() if x.is_dir()]

    sequences = []
    for subdir in subdirs:
        sequence = []
        files = [x for x in subdir.iterdir() if x.is_file()]
        #Se ordenan los archivos para cargar las secuencias en orden.
        files = sorted( files, key=lambda x:int( ''.join(list( filter(str.isdigit, str(Path(x).stem)) )) ) )

        for file in files:
            sequence.append(load_img(str(file)))

        sequences.append(sequence)
    return sequences

real_path = Path("/home/dadepe/Documentos/faces/real")
fake_path = Path("/home/dadepe/Documentos/faces/fake")

real_sequences = load_sequences(real_path)
fake_sequences = load_sequences(fake_path)

print("Se han cargado {} directorios reales y {} falsos.".format(len(real_sequences),len(fake_sequences)))