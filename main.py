import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *


list = []

model=YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('vidyolov8.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count=0

area = [(270, 238), (285, 262), (592, 226), (552, 207)]
tracker = Tracker()
area_c = set()

while True:
    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))

    list.clear()

    results = model.predict(frame)

    # Extrair coordenadas xyxy e classes
    boxes = results[0].boxes.xyxy
    classes = results[0].boxes.cls

    # Criar um DataFrame com as coordenadas xyxy e classes
    df = pd.DataFrame({
        'x1': boxes[:, 0],
        'y1': boxes[:, 1],
        'x2': boxes[:, 2],
        'y2': boxes[:, 3],
        'class': [model.names[int(c)] for c in classes]
    })

    # Desenhar as caixas delimitadoras e as classes no frame
    for index, row in df.iterrows():
        x1, y1, x2, y2, c = row

        if 'car' in c:
            list.append([x1, y1, x2, y2])
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx = int(x3 + x4) // 2
            cy = int(y3 + y4) // 2
            results = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)

            if results >= 0:
                cv2.circle(frame, (cx, cy), 4, (255,191,117), -1)
                cv2.rectangle(frame, (int(x3), int(y3)), (int(x4), int(y4)), (255,191,117), 2)
                cv2.putText(frame, str(id), (int(x3), int(y3)), cv2.FONT_HERSHEY_COMPLEX, 0.7, (117,158,255), 1)
                area_c.add(id)

    cv2.polylines(frame, [np.array(area, np.int32)], True, (255,164,58), 3)
    text= "Carros: " + str(len(area_c))
    cv2.putText(frame, text, (67, 147), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()