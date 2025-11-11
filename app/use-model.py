import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib

model = joblib.load("emotion_model.pkl")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
drawing = mp.solutions.drawing_utils

meshid = [10, 152, 234, 454, 323, 93, 132, 361, 389, 15, 16,33, 133, 159, 145,362, 263, 386, 374,70, 63, 105, 107,336, 296, 334, 300,1, 2, 4, 5, 6, 168,61, 291, 0, 17, 13, 14, 78, 308, 82, 312, 87, 317, 95, 321, 185, 410, 152, 199, 200, 172, 175, 378,159,145]

# Load input image
image_path = "test_image.jpg"  # <- Change to your image file
image = cv.imread(image_path)
if image is None:
    print("Image not found. Check the path.")
    exit()

# Convert to RGB for mediapipe
rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
result = face_mesh.process(rgb_image)

if not result.multi_face_landmarks:
    print("No face detected in the image.")
    exit()

for landmarks in result.multi_face_landmarks:
    h, w, _ = image.shape
    current = {}
    for id in meshid:
        x = int(landmarks.landmark[id].x * w)
        y = int(landmarks.landmark[id].y * h)
        current[id] = (x, y)
        cv.circle(image, (x, y), 2, (0, 255, 0), -1)

    def dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    features = {
        "mouth_open": dist(current[0], current[17]),
        "eye_gap": dist(current[159], current[145]),
        "mouth_width": dist(current[61], current[291]),
        "mouth_curve": ((current[61][1] + current[291][1]) / 2) - current[0][1]
    }

    X_input = pd.DataFrame([features])
    emotion = model.predict(X_input)[0]

    cv.putText(image, f"Emotion: {emotion}", (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    print(f"Predicted Emotion: {emotion}")

# Show image
cv.imshow("Emotion Detection", image)
cv.waitKey(0)
cv.destroyAllWindows()
