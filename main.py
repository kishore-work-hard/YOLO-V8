import json
import cv2
from ultralytics import YOLO
import pandas as pd

model = YOLO("nano.pt")

def get_od(img):
    x = model.predict(source=img, show=False)
    bbox_pd = pd.DataFrame(x[0].boxes.xyxy).astype("float")
    cls_pd = pd.DataFrame(x[0].boxes.cls).astype("float")

    car_boxes = [
        [int(row[0]), int(row[1]), int(row[2]), int(row[3])]
        for index, row in bbox_pd.iterrows() if int(cls_pd.loc[index, 0]) == 0
    ]

    return car_boxes

def main():
    cams = json.load(open('example.json'))

    while True:
        for cam_ip, cam in cams.items():
            print(cam_ip)
            img = cv2.imread('car1.jpg')
            o = {'area': cam_ip, 'spots': []}
            bbox = get_od(img)
            print(bbox)

if __name__ == "__main__":
    main()
