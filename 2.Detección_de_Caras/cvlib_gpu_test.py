#encoding: utf-8
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import cvlib as cv
import cv2
from pathlib import Path
import time
import json

tic = time.perf_counter()

frames_directory = "./test_frames"
subdirs = [x for x in Path(frames_directory).iterdir() if x.is_dir()]

pairs = {}
for idx,subdir in enumerate(subdirs):
    print("Procesando el directorio {}. {}/{}".format(str(subdir),idx+1, len(subdirs)))
    for image_path in subdir.glob('*.jpg'):    
        faces = cv.detect_face(cv2.imread(str(image_path)),enable_gpu=True) 
        corrected_faces = []
        for face in faces[0]:
            corrected_face = []
            for num in face:
               corrected_face.append(num.item())
            corrected_faces.append(corrected_face)
        pairs[str(image_path)] = corrected_faces

with open("real-faces.json", "w") as write_file:
    json.dump(pairs, write_file)

toc = time.perf_counter()

print("Se procesaron {} imágenes.".format(len(pairs)))
print("Se tardó {:0.4f} segundos.".format(toc - tic))