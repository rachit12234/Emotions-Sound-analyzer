# IMPORTING LIBRARIES
import cv2 as cv
import mediapipe as mp

# SETTING MEDIAPIPE
face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
faceMesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5)
drawing = mp.solutions.drawing_utils
# SETTING CAMERA
webcam = cv.VideoCapture(0)

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
            drawing.draw_landmarks(frame,landmarks)

    frame = cv.cvtColor(frame,cv.COLOR_RGB2BGR)

    cv.imshow("camera",frame)

    if cv.waitKey(5) & 0xFF == ord("q"):
        break