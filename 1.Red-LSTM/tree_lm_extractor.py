import lm_extractor_FAN
import argparse
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, help="Input folder.", required=True)
parser.add_argument('--op', type=str, help="Output location.", required=True)
args = parser.parse_args()

init_time = time.clock()

input_path = Path(args.ip)
subdirs = [x for x in input_path.iterdir() if x.is_dir()]

output_path = Path(args.op)

for index,subdir in enumerate(subdirs):
    print('Extrayendo landmarks de la carpeta ' + str(index) + '/' + str(len(subdirs)))
    lm_extractor_FAN.landmark_from_dir(subdir, output_path, subdir.name)
    print('Landmarks extracted for ' + subdir.name)

end_time = time.clock()
print('Tiempo total ' + str(end_time - init_time) + 's')