import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('vids', type=str, nargs='+', help="Input videos.")
parser.add_argument('--op', type=str, required=True, help="Output location.")
parser.add_argument('--dur', type=str, required=True, help="Duration of extraction.")
args = parser.parse_args()

for vid in args.vids:
    folder_name = os.path.splitext(os.path.basename(vid))[0]
    folder_path = args.op + '/' + folder_name

    vidcap = cv2.VideoCapture(vid)
    success,image = vidcap.read()
    count = 0

    fps = 25

    total_frames = fps * int(args.dur)

    try:
        os.mkdir(folder_path)
    except OSError:
        print ("Creation of the directory %s failed" % folder_path)
    else:
        print ("Successfully created the directory %s " % folder_path)

    while success and count <= total_frames:
        cv2.imwrite(folder_path + "/frame%d.jpg" % count, image)  
        success,image = vidcap.read()

        count += 1