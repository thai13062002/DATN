import sys
import cv2
import numpy as np
import dlib
import math
import vlc
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from imutils import face_utils
from ultralytics import YOLO
import torch
import pythoncom

# Initialize COM for VLC
pythoncom.CoInitialize()

def yawn(mouth):
    return ((euclideanDist(mouth[2], mouth[10]) + euclideanDist(mouth[4], mouth[8])) / (2 * euclideanDist(mouth[0], mouth[6])))

def getFaceDirection(shape, size):
    image_points = np.array([
        shape[33],    # Nose tip
        shape[8],     # Chin
        shape[45],    # Left eye left corner
        shape[36],    # Right eye right corner
        shape[54],    # Left Mouth corner
        shape[48]     # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    return translation_vector[1][0]

def euclideanDist(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))

def ear(eye):
    return ((euclideanDist(eye[1], eye[5]) + euclideanDist(eye[2], eye[4])) / (2 * euclideanDist(eye[0], eye[3])))

class VideoThread(QTimer):
    def __init__(self, label, model, detector, predictor, leStart, leEnd, reStart, reEnd, mStart, mEnd, classNames):
        super().__init__()
        self.label = label
        self.capture = cv2.VideoCapture(0)
        self.model = model
        self.detector = detector
        self.predictor = predictor
        self.leStart = leStart
        self.leEnd = leEnd
        self.reStart = reStart
        self.reEnd = reEnd
        self.mStart = mStart
        self.mEnd = mEnd
        self.classNames = classNames
        self.alert = vlc.MediaPlayer('take_focus.mp3')
        self.alert2 = vlc.MediaPlayer('take_focus.mp3')
        self.take_break = vlc.MediaPlayer('take_break.mp3')
        self.alarm = vlc.MediaPlayer('alarm-6786.mp3')
        self.frame_thresh_1 = 5
        self.frame_thresh_2 = 10
        self.close_thresh = 0.23
        self.flag = 0
        self.yawn_countdown = 0
        self.distracted_countdown = 0
        self.head_count = 0
        self.avgEAR = 0
        self.display_frame_flag = True

        self.timeout.connect(self.update_frame)
        self.start(20)

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return

        size = frame.shape
        results = self.model(frame, verbose=False)
        object_detection = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                object_detection.append(self.classNames[cls])
                org = [x1, y1-30]
                cv2.putText(frame, self.classNames[cls], org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        gray = frame.copy()
        rects = self.detector(gray, 0)
        if len(rects):
            shape = face_utils.shape_to_np(self.predictor(gray, rects[0]))
            leftEye = shape[self.leStart:self.leEnd]
            rightEye = shape[self.reStart:self.reEnd]
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(shape[self.mStart:self.mEnd])
            leftEAR = ear(leftEye)
            rightEAR = ear(rightEye)
            self.avgEAR = (leftEAR + rightEAR) / 2.0
            eyeContourColor = (255, 255, 255)

            if (yawn(shape[self.mStart:self.mEnd]) > 0.6) or ('yawn' in object_detection):
                cv2.putText(gray, "Yawn Detected", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                self.yawn_countdown += 1
                if self.yawn_countdown > 15:
                    self.take_break.play()
            elif 'yawn' not in object_detection and self.yawn_countdown:
                self.yawn_countdown = 0
                self.take_break.stop()

            if self.avgEAR < self.close_thresh or 'drowsy' in object_detection:
                self.flag += 1
                eyeContourColor = (0, 255, 255)
                if self.flag >= self.frame_thresh_1 and getFaceDirection(shape, size) < 0:
                    eyeContourColor = (255, 0, 0)
                    cv2.putText(gray, "Drowsy (Body Posture)", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                    self.alert.play()
                if self.flag >= self.frame_thresh_2:
                    eyeContourColor = (147, 20, 255)
                    cv2.putText(gray, "Drowsy", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                    self.alarm.play()
            elif self.avgEAR > self.close_thresh and self.flag:
                self.alert.stop()
                self.alarm.stop()
                self.flag = 0

            cv2.drawContours(gray, [leftEyeHull], -1, eyeContourColor, 2)
            cv2.drawContours(gray, [rightEyeHull], -1, eyeContourColor, 2)
            cv2.drawContours(gray, [mouthHull], -1, eyeContourColor, 2)

        if any(objects in ['phone', 'smoking'] for objects in object_detection):
            self.distracted_countdown += 1
            if self.distracted_countdown > 10:
                self.alert2.play()
        elif self.distracted_countdown:
            self.distracted_countdown = 0
            self.alert2.stop()

        if any(objects in ['head drop', 'distracted'] for objects in object_detection) or (len(object_detection) == 0 and len(rects) ==0):
            self.head_count += 1
            if self.head_count > 5:
                self.alert2.play()
                self.alarm.play()
        elif self.head_count:
            self.head_count = 0
            self.alert2.stop()
            self.alarm.stop()

        if self.display_frame_flag:
            self.display_frame(gray)

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(q_image))
    def stop(self):
        self.capture.release()
        self.display_frame_flag=False
    def stopTimer(self):
        self.stop()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Driver Monitoring System")
        self.setGeometry(100, 100, 800, 600)
        self.video_thread = None

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.label = QLabel(self)
        self.layout.addWidget(self.label)

        self.yolo_button = QPushButton('Upload Image to Detect', self)
        self.yolo_button.clicked.connect(self.load_and_detect_yolo)
        self.layout.addWidget(self.yolo_button)

        self.detect_button = QPushButton('Start Detection Webcam', self)
        self.detect_button.clicked.connect(self.start_detection)
        self.layout.addWidget(self.detect_button)
        
        self.stop_button = QPushButton('Stop Detection Webcam', self)
        self.stop_button.clicked.connect(self.stop_detection)
        self.layout.addWidget(self.stop_button)

        self.model = YOLO('./yolov8/weights/best.pt').to('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.leStart, self.leEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.reStart, self.reEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.mStart, self.mEnd = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        self.classNames = ['awake', 'distracted', 'drowsy', 'head drop', 'phone', 'smoking', 'yawn']

    def load_and_detect_yolo(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;JPEG (*.jpg;*.jpeg);;PNG (*.png)", options=options)
        if file_name:
            image = cv2.imread(file_name)
            image = cv2.resize(image, (640, 640))
            results = self.model(image)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(image, self.classNames[cls], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            self.display_image(image)

    def display_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(q_image))

    def start_detection(self):
        if self.video_thread is None:
            self.video_thread = VideoThread(self.label, self.model, self.detector, self.predictor, self.leStart, self.leEnd, self.reStart, self.reEnd, self.mStart, self.mEnd, self.classNames)
        else:
            print("Detection is already running")
        
    def stop_detection(self):
        if self.video_thread:
            self.video_thread.stopTimer()
            self.video_thread.stop()
            self.video_thread = None
            print("Stop Detection")
        else:
            print("No detection running")
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
