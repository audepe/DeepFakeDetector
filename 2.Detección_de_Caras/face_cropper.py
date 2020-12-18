import json
import time
import cv2
from pathlib import Path
from tqdm import tqdm

with open("real-faces-sorted.json", "r") as read_file:
    data = json.load(read_file)

wrong_images = []
for key in tqdm(data.keys()):
    file_path = Path(key)
    output_file = Path(str(file_path.parent) + "/" + str(file_path.stem) + "_face.jpg")

    if output_file.is_file():
        continue

    original_image = cv2.imread(str(file_path))
    (x,y,j,k) = data[key][0]
    crop_img = original_image[y:k, x:j]
    cv2.imwrite(str(output_file), crop_img)