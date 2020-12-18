import json
import cv2

with open("fake-faces.json", "r") as read_file:
    data = json.load(read_file)

sample_key = list(data.keys())[3]
sample_face = data[sample_key][0]

img = cv2.imread(sample_key)
(x,y,w,h) = sample_face
cv2.rectangle(img, (x, y), (w, h), (255, 0, 0), 2)

cv2.imshow("rectangled", img)
cv2.waitKey(0)