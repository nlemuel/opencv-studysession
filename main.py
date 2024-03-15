import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO



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
while True:
    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))

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
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,191,117), 2)
        cv2.putText(frame, str(c), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, 0.7, (117,158,255), 1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()