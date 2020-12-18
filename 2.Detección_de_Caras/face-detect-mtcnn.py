from mtcnn import MTCNN
from pathlib import Path
import cv2

detector = MTCNN()
for image_path in Path("/media/dadepe/Elements/frames/real/000/").glob('*.jpg'):
    
    detector.detect_faces(cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB))
