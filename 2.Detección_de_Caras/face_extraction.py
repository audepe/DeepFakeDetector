import dlib
import cv2
from pathlib import Path


def landmark_from_dir(frames_directory, output_dir):
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    images = []
    for image_path in Path(str(frames_directory)).glob('*.jpg'):        
        images.append(cv2.imread(str(image_path)))

    print(len(images))
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, help="Input folder.", required=True)
    parser.add_argument(
        '--op', type=str, help="Output location.", required=True)
    args = parser.parse_args()

    landmark_from_dir(args.ip, args.op)


if __name__ == "__main__":
    main()
