import time
import random
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from deepface import DeepFace 
backends = ['opencv', 'ssd', 'dlib', 'mtcnn']

frames_directory = "./test_frames"
subdirs = [x for x in Path(frames_directory).iterdir() if x.is_dir()]
#subdirs = random.sample(subdirs,20)

for backend in backends:
    print("Probando {}".format(backend))
    processed = 0
    tic = time.perf_counter()
    for idx,subdir in enumerate(subdirs):
        for image_path in subdir.glob('*.jpg'):    
            try:
                detected_aligned_face = DeepFace.detectFace(img_path = str(image_path) , detector_backend = backend)
                processed += 1
            except:
                continue

    toc = time.perf_counter()
    print("Se procesaron {} imágenes.".format(processed))
    print("Se tardó {:0.4f} segundos.".format(toc - tic))