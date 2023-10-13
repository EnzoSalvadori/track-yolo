import os
import cv2
import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

video_path = os.path.join('.', 'carro_teste - Trim.mp4')
video_out_path = os.path.join('.', 'out_teste.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
#frame = frame[180:980,200:1500]

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (640, 640))

model = YOLO("yolov8n.pt")
detection_threshold = 0.5

object_tracker = DeepSort(max_age=5)

id_cout = []

while ret:

    ret, frame = cap.read()
    frame = cv2.resize(frame,(640,640))
    #frame = frame[180:980,700:1500]

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
                detections.append(([x1, y1, (x2-x1),(y2-y1)],score,class_id))

    tracks = object_tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        bbox = ltrb

        if len(id_cout) > int(track_id):
            id_cout[int(track_id)] += 1
            if id_cout[int(track_id)] >= 5:
                filename = "fotos/"+(
                        str(datetime.datetime.now())
                        .replace(" ", "")
                        .replace("-", "")
                        .replace(":", "")
                        .replace(".", "")
                    +"_"+str(track_id)+".jpg")
                print(int(bbox[0]))
                print(int(bbox[1]))
                print(int(bbox[2]))
                print(int(bbox[3]))
                print(frame.shape)
                img = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                #cv2.imwrite(filename,img)
                cv2.rectangle(frame,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255),2)
                cv2.putText(frame, "ID " + str (track_id),(int(bbox[0]),int(bbox[1]-10)),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
        else:
            id_cout.append(1)

    cap_out.write(frame)
    cv2.imshow('WebCan', frame)
    ret, frame = cap.read()

    if cv2.waitKey(1) == 27:
        break

cap.release()
cap_out.release()
cv2.destroyAllWindows()