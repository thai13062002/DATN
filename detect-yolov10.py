from ultralytics import YOLOv10
import cv2

model = YOLOv10('./yolov10_model/weights/best.pt')
print(model.info())
# result = model.predict(source="0", show=True)
# print(result)