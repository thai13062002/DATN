import cv2
import math
import numpy as np
import dlib
import imutils
from imutils import face_utils
from matplotlib import pyplot as plt
import vlc
# import train as train
import sys, webbrowser, datetime
from ultralytics import YOLO
import torch

def yawn(mouth):
    return ((euclideanDist(mouth[2], mouth[10])+euclideanDist(mouth[4], mouth[8]))/(2*euclideanDist(mouth[0], mouth[6])))

def getFaceDirection(shape, size):
    image_points = np.array([
                                shape[33],    # Nose tip
                                shape[8],     # Chin
                                shape[45],    # Left eye left corner
                                shape[36],    # Right eye right corne
                                shape[54],    # Left Mouth corner
                                shape[48]     # Right mouth corner
                            ], dtype="double")
    
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            
                            ])
    
    # Camera internals
    
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    return(translation_vector[1][0])

def euclideanDist(a, b):
    return (math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2)))

#EAR -> Eye Aspect ratio
def ear(eye):
    return ((euclideanDist(eye[1], eye[5])+euclideanDist(eye[2], eye[4]))/(2*euclideanDist(eye[0], eye[3])))

def writeEyes(a, b, img):
    y1 = max(a[1][1], a[2][1])
    y2 = min(a[4][1], a[5][1])
    x1 = a[0][0]
    x2 = a[3][0]
    cv2.imwrite('left-eye.jpg', img[y1:y2, x1:x2])
    y1 = max(b[1][1], b[2][1])
    y2 = min(b[4][1], b[5][1])
    x1 = b[0][0]
    x2 = b[3][0]
    cv2.imwrite('right-eye.jpg', img[y1:y2, x1:x2])

alert = vlc.MediaPlayer('take_focus.mp3')
alert2 = vlc.MediaPlayer('take_focus.mp3')

take_break = vlc.MediaPlayer('take_break.mp3')
alarm = vlc.MediaPlayer('alarm-6786.mp3')

frame_thresh_1 = 5
frame_thresh_2 = 10

close_thresh = 0.23#(close_avg+open_avg)/2.0
flag = 0
yawn_countdown = 0
distracted_countdown = 0
head_count = 0
map_flag = 1

# print(close_thresh)

capture = cv2.VideoCapture(0)
avgEAR = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('./yolov8/weights/best.pt').to(device)
classNames = ['awake','distracted','drowsy','head drop','phone','smoking','yawn']

phone_smoke = ['phone','smoking']
headDrop = ['head drop','distracted']

while(True):
    ret, frame = capture.read()
    size = frame.shape
    # YOLOv8 object detection
    results = model(frame, verbose=False)
    object_detection = []
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            # confidence = math.ceil((box.conf[0]*100))/100
            # print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            object_detection.append(classNames[cls])
            # print("Class name -->", classNames[cls])
            # print('--------------------------------')

            # object details
            org = [x1, y1-30]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 1

            cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)
    gray = frame.copy()
    rects = detector(gray, 0)
    if(len(rects)):
        shape = face_utils.shape_to_np(predictor(gray, rects[0]))
        leftEye = shape[leStart:leEnd]
        rightEye = shape[reStart:reEnd]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(shape[mStart:mEnd])
        # print("Mouth Open Ratio", yawn(shape[mStart:mEnd]))
        leftEAR = ear(leftEye) #Get the left eye aspect ratio
        rightEAR = ear(rightEye) #Get the right eye aspect ratio
        avgEAR = (leftEAR+rightEAR)/2.0
        eyeContourColor = (255, 255, 255)

        if((yawn(shape[mStart:mEnd])>0.6) or ('yawn' in object_detection)):
            cv2.putText(gray, "Yawn Detected", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
            yawn_countdown+=1
            print(yawn_countdown)
            if yawn_countdown > 15:
                take_break.play()
        elif(('yawn' not in object_detection) and yawn_countdown):
            print('yawn reset 0')
            yawn_countdown=0
            take_break.stop()

        if(avgEAR<close_thresh) or ('drowsy' in object_detection):
            flag+=1
            eyeContourColor = (0,255,255)
            print(flag)
            if(flag>=frame_thresh_1 and getFaceDirection(shape, size)<0):
                eyeContourColor = (255, 0, 0)
                cv2.putText(gray, "Drowsy (Body Posture)", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                alert.play()          
                if(map_flag):
                    map_flag = 0
            if(flag>=frame_thresh_2):
                eyeContourColor = (147, 20, 255)
                cv2.putText(gray, "Drowsy", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                alarm.play()          
                if(map_flag):
                    map_flag = 0 
        elif(avgEAR>close_thresh and flag):
            print("Flag reseted to 0")
            alert.stop()
            alarm.stop()
            map_flag=1
            flag=0

        cv2.drawContours(gray, [leftEyeHull], -1, eyeContourColor, 2)
        cv2.drawContours(gray, [rightEyeHull], -1, eyeContourColor, 2)
        cv2.drawContours(gray, [mouthHull],-1, eyeContourColor,2)
    # if(avgEAR>close_thresh):
    #     alert.stop()
    
    if any(objects in phone_smoke for objects in object_detection):
        distracted_countdown+=1
        print('distracted: ',distracted_countdown)
        if distracted_countdown > 10:
            alert2.play()
    elif(distracted_countdown):
        print('distract reset 0')
        distracted_countdown=0
        alert2.stop()

    if any(objects in headDrop for objects in object_detection) or (len(object_detection)==0):
        head_count+=1
        print('head: ',head_count)
        if head_count > 5:
            alert2.play()
            alarm.play()
    elif(head_count):
        print('head reset 0')
        head_count=0
        alert2.stop()
        alarm.stop()
        
    # combined_frame = cv2.addWeighted(frame, 0.5, gray, 0.5, 0)
    cv2.imshow('Driver', gray)
    
    if(cv2.waitKey(1)==27):
        break
        
capture.release()
cv2.destroyAllWindows()