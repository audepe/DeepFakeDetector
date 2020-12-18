import json
import numpy
from pathlib import Path

with open("fake-faces-sorted.json", "r") as read_file:
    data = json.load(read_file)

sorted_keys = sorted(data.keys(), key=lambda x:x.lower() ) 
sorted_dict = {}

for key in sorted_keys:
        sorted_dict[key] = data[key]

directories = []
for key in sorted_keys:
    directories.append(str(Path(key).parents[0]))

directories_set = set(directories)

ordered_dict = {}
for directory in directories_set:
    result = [key for key in data.keys() if key.startswith(directory)]
    sorted_dir = sorted( result, key=lambda x:int( ''.join(list( filter(str.isdigit, str(Path(x).stem)) )) ) )
    if len(sorted_dir) != 251:
        continue
    for key in sorted_dir:
        ordered_dict[key] = data[key]

print("El diccionario a guardar tiene {} entradas.".format(len(ordered_dict)))

with open("fake-faces-sorted.json", "w") as write_file:
    json.dump(ordered_dict, write_file)