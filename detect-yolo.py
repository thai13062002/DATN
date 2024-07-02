from ultralytics import YOLO
import cv2

model = YOLO('./yolov8/weights/best.pt')
print(model)
# result = model.predict(source="0", show=True)
# print(result)