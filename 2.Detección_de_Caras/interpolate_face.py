import json
import numpy as np
from pathlib import Path


def get_mid_face(a, b):
    c = np.add(a, b)
    c = np.divide(c, 2)
    c = c.astype(int)
    c = c.tolist()
    for num in c:
        if num < 0:
            return None
    return c


with open("real-faces-sorted.json", "r") as read_file:
    data = json.load(read_file)

completed_dict = {}

for idx, key in enumerate(list(data.keys())):
    if len(data[key]) == 0:
        if idx > 0 and idx < len(data.keys()):
            previous_element = data[list(data.keys())[idx-1]]
            next_element = data[list(data.keys())[idx+1]]
            if len(previous_element) == len(next_element):
                missing_faces = []
                for prev_face, next_face in zip(previous_element, next_element):
                    missing_face = get_mid_face(prev_face, next_face)
                    if missing_face == None:
                        break
                    missing_faces.append(missing_face)
                if len(missing_faces) == 0:
                    continue
                for face in missing_faces:
                    if face == None:
                        continue
                data[key] = missing_faces
            else:
                continue
        else:
            continue
    else:
        counted = 0
        for face in data[key]:
            for element in face:
                if element < 0:
                    counted = counted + 1
        if counted > 0:
            continue
    completed_dict[key] = data[key]

print("The original data had {} entries and the stripped down has {}.".format( len(data), len(completed_dict) ))
with open("real-faces-sorted.json", "w") as write_file:
    json.dump(completed_dict, write_file)