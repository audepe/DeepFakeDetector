import face_alignment
from pathlib import Path
import time
import json

def landmark_from_dir(frames_directory, output_dir, output_filename):    
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    output_file = output_dir.joinpath(str(output_filename))
    output_file = output_file.with_suffix('.json')

    if output_file.is_file():
        print(str(output_file) + ' ya existe.')
        return

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

    preds = fa.get_landmarks_from_directory(str(frames_directory))

    for key, value in preds.items():
        if preds[key] == None:
            f = open(output_file,"w")
            f.write("No landmarks.")
            f.close()
            return

        preds[key] = value[0].tolist()

    with open(output_file, "w") as write_file:
        json.dump(preds, write_file)

def main():
    init_time = time.clock()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, help="Input folder.", required=True)
    parser.add_argument('--op', type=str, help="Output location.", required=True)
    args = parser.parse_args()

    landmark_from_dir(args.ip, args.op, 'test')

    end_time = time.clock()
    print('Tiempo total ' + str(end_time - init_time) + 's')

if __name__ == "__main__":
    main()