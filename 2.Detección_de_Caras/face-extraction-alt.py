import cv2
from fdet import io, RetinaFace

BATCH_SIZE = 10

detector = RetinaFace(backbone='MOBILENET', cuda_devices=[0])
vid_cap = cv2.VideoCapture('test_video.mp4')

video_face_detections = []  # list to store all video face detections
image_buffer = []  # buffer to store the batch

while True:

    success, frame = vid_cap.read()  # read the frame from video capture
    if not success:
        break  # end of video

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB
    image_buffer.append(frame)  # add frame to buffer

    if len(image_buffer) == BATCH_SIZE:  # if buffer is full, detect the batch
        batch_detections = detector.batch_detect(image_buffer)
        video_face_detections.extend(batch_detections)
        image_buffer.clear()  # clear the buffer

if image_buffer:  # checks if images remain in the buffer and detect it
    batch_detections = detector.batch_detect(image_buffer)
    video_face_detections.extend(batch_detections)
