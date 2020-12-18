import json
from pathlib import Path

root_path = Path("../faces_resized")
fake_path = root_path.joinpath("fake")
real_path = root_path.joinpath("real")

fake_subdirs = [x for x in fake_path.iterdir() if x.is_dir()]
real_subdirs = [x for x in real_path.iterdir() if x.is_dir()]

sequences_data = {}
for fake_subdir in fake_subdirs:
    sequences_data[str(fake_subdir)] = 1

for real_subdir in real_subdirs:
    sequences_data[str(real_subdir)] = 0

with open("sequences.json", "w") as write_file:
    json.dump(sequences_data, write_file)