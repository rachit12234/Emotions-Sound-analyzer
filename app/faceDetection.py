# IMPORTING LIBRARIES
import cv2 as cv
import mediapipe as mp
import numpy as np
from collections import deque, Counter

# SETTING MEDIAPIPE
face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
faceMesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5)
drawing = mp.solutions.drawing_utils
# SETTING CAMERA
webcam = cv.VideoCapture(0)

meshid = [10, 152, 234, 454, 323, 93, 132, 361, 389, 15, 16,33, 133, 159, 145,362, 263, 386, 374,70, 63, 105, 107,336, 296, 334, 300,1, 2, 4, 5, 6, 168,61, 291, 0, 17, 13, 14, 78, 308, 82, 312, 87, 317, 95, 321, 185, 410, 78, 308,152, 199, 200, 172, 175, 378,159,145] 

cal ={}
i = 0
emotion_history = deque(maxlen=8)
calibrated = False

def detect_emotion(current, cal):

    def mouth_open():
        cc0 = current[0]
        ca0 = cal[0]
        cc17 = current[17]
        ca17 = cal[17]

        calibrated_dist = abs(ca0[1] - ca17[1])
        current_dist = abs(cc0[1] - cc17[1])

        if calibrated_dist < current_dist:
            return True
        else:
            return False

    def eyes_closed():
        cc159 = current[159]   # left eye top
        cc145 = current[145]   # left eye bottom
        ca159 = cal[159]
        ca145 = cal[145]

        calibrated_dist = abs(ca159[1] - ca145[1])
        current_dist = abs(cc159[1] - cc145[1])

        if current_dist < (calibrated_dist * 0.5):
            return True
        else:
            return False

    def frown():
        # Mouth corners: 61 (left), 291 (right)
        # Mouth center: 0 (or 13/14 can be used)
        cc61 = current[61]
        cc291 = current[291]
        cc0 = current[0]

        ca61 = cal[61]
        ca291 = cal[291]
        ca0 = cal[0]

        # Average mouth corner height (Y)
        cal_corners_y = (ca61[1] + ca291[1]) / 2
        cur_corners_y = (cc61[1] + cc291[1]) / 2

        # Compare with mouth center
        cal_center_y = ca0[1]
        cur_center_y = cc0[1]

        # Calculate vertical difference
        cal_diff = cal_corners_y - cal_center_y
        cur_diff = cur_corners_y - cur_center_y

        # If corners go down more (difference decreases)
        if cur_diff < cal_diff * 0.8:
            return True
        else:
            return False

    # ----------- Main Logic -----------

    if mouth_open() and not eyes_closed():
        return "Smile"
    elif mouth_open() and eyes_closed():
        return "Laugh"
    elif frown() and not eyes_closed():
        return "Sad"
    elif frown() and eyes_closed():
        return "Cry"
    else:
        return "Normal"


while webcam.isOpened() :
    success, frame = webcam.read()

    frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results = face.process(frame)
    mesh_result = faceMesh.process(frame)

    if results.detections:
        for detection in results.detections:
            drawing.draw_detection(frame,detection)

    if mesh_result.multi_face_landmarks:
        for landmarks in mesh_result.multi_face_landmarks :
            #drawing.draw_landmarks(frame,landmarks)

            height , width  , _ = frame.shape
            current = {}
            for id in meshid:
                
                x = int((landmarks.landmark[id].x)*width)
                y = int((landmarks.landmark[id].y)*height)
                current[id] = (x, y)
                cv.circle(frame, (x, y), 3, (0, 255, 0), -1)
                cv.putText(frame, str(id), (x+4, y-4), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

                # cal[i] = (x, y)
                # i = i+1

            if calibrated:
                emotion = detect_emotion(current, cal)
                emotion_history.append(emotion)
                smoothed = Counter(emotion_history).most_common(1)[0][0]
                cv.putText(frame, f"Emotion: {smoothed}", (30, 50),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
            

    cv.putText(frame, "Press T to calibrate", (20, 30),cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    frame = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
   

    cv.imshow("camera", frame)
            
    key = cv.waitKey(5) & 0xFF
    if key == ord("t"):
        # print("Calibration snapshot:")
        # cal = {id: (int(landmarks.landmark[id].x * width),int(landmarks.landmark[id].y * height)) for id in meshid}
        # calibrated = True
        # print("✅ Calibration captured for one frame:")
        # print(cal)
        cal = {id: current[id] for id in meshid}
        calibrated = True
        print("✅ Calibration snapshot captured!")

    elif key == ord("q"):
        break
    
webcam.release()
cv.destroyAllWindows