import os
import random

import cv2
from ultralytics import YOLO

from tracker import Tracker

#parametros
USUARIO = "admin"
SENHA = "centro12"
IP = "192.168.1.24"
PORTA = "554"
CANAL = "1"

URL = 'rtsp://{}:{}@{}:{}/cam/realmonitor?channel={}&subtype=0'.format(USUARIO,SENHA,IP,PORTA,CANAL)
print(f'conectado com: {URL}')

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

cap = cv2.VideoCapture(URL,cv2.CAP_FFMPEG)
ret, frame = cap.read()

model = YOLO("yolov8n.pt")

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5

while True:

    print(frame.shape)

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

    cv2.imshow('WebCan', frame)
    ret, frame = cap.read()

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()