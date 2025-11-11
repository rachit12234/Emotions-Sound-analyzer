import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5)

meshid = [10, 152, 234, 454, 323, 93, 132, 361, 389, 15, 16, 33, 133, 159, 145,
          362, 263, 386, 374, 70, 63, 105, 107, 336, 296, 334, 300, 1, 2, 4, 5, 6,
          168, 61, 291, 0, 17, 13, 14, 78, 308, 82, 312, 87, 317, 95, 321, 185,
          410, 78, 308, 152, 199, 200, 172, 175, 378, 159, 145]

def extract_landmarks(image_path):
    image = cv.imread(image_path)
    if image is None:
        return None
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0]
    h, w, _ = image.shape
    coords = {}
    for id in meshid:
        x = int(landmarks.landmark[id].x * w)
        y = int(landmarks.landmark[id].y * h)
        coords[id] = (x, y)
    return coords


def compute_features(current):
    features = {}

    mouth_open = abs(current[0][1] - current[17][1])
    features["mouth_open"] = mouth_open

    eye_open = abs(current[159][1] - current[145][1])
    features["eye_open"] = eye_open

    mouth_center = current[0][1]
    corners_avg = (current[61][1] + current[291][1]) / 2
    features["frown_ratio"] = corners_avg - mouth_center

    return features

DATASET_DIR = "dataset"

data = []
for label in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(folder_path):
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        lm = extract_landmarks(img_path)
        if lm is not None:
            feats = compute_features(lm)
            feats["label"] = label
            data.append(feats)
            print(f"Processed {img_name} for {label}")

df = pd.DataFrame(data)
df.to_csv("emotion_features.csv", index=False)
print("Feature extraction complete! Saved as emotion_features.csv")