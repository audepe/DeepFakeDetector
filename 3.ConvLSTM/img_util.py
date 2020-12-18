import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import cv2
from tqdm import tqdm

def load_sequences(path):
    subdirs = [x for x in path.iterdir() if x.is_dir()]

    sequences = []
    for subdir in tqdm(subdirs[:500]):
        sequence = []
        files = [x for x in subdir.iterdir() if x.is_file()]
        # Se ordenan los archivos para cargar las secuencias en orden.
        files = sorted(files, key=lambda x: int(
            ''.join(list(filter(str.isdigit, str(Path(x).stem))))))

        for file in files:
            sequence.append(np.asarray(load_img(str(file))))
        
        if len(sequence) != 251:
            print(str(subdir))
            continue
        sequences.append(np.asarray(sequence))
    return sequences

def resize_and_save(path):
    subdirs = [x for x in path.iterdir() if x.is_dir()]

    for subdir in tqdm(subdirs):
        files = [x for x in subdir.iterdir() if x.is_file()]
        # Se ordenan los archivos para cargar las secuencias en orden.
        for file in files:
            output_file = str(file.parent) + "/" + file.stem + "_test.jpg"
            if Path(output_file).is_file():
                continue
            load_img(str(file)).resize((64,64), Image.ANTIALIAS).save(output_file, "JPEG")


def get_max_size(sequences):
    max_x = 0
    max_y = 0
    for sequence in sequences:
        for im in sequence:
            if im.size[0] > max_x:
                max_x = im.size[0]
            if im.size[1] > max_y:
                max_y = im.size[1]
    return max_x, max_y


def get_min_size(sequences):
    min_x = 9999
    min_y = 9999
    for sequence in sequences:
        for im in sequence:
            if im.size[0] < min_x:
                min_x = im.size[0]
            if im.size[1] < min_y:
                min_y = im.size[1]
    return min_x, min_y


def print_sizes(real_sequences, fake_sequences):
    max_real_size = get_max_size(real_sequences)
    max_fake_size = get_max_size(fake_sequences)

    min_real_size = get_min_size(real_sequences)
    min_fake_size = get_min_size(fake_sequences)

    print("El tamaño máximo de los reales es {} y de los fakes es {}.".format(max_real_size,max_fake_size))
    print("El tamaño mínimo de los reales es {} y de los fakes es {}.".format(min_real_size,min_fake_size))

