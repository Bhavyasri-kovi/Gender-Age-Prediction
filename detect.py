import cv2
import numpy as np

AGE_MODEL = "models/age_net.caffemodel"
AGE_PROTO = "models/age_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"
GENDER_PROTO = "models/gender_deploy.prototxt"

AGE_BUCKETS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]
GENDER_BUCKETS = ["Male", "Female"]

# Load pretrained CNN models
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

def detect_age_gender(face_frame):
    blob = cv2.dnn.blobFromImage(face_frame, 1.0, (227, 227),
                                 (78.426, 87.769, 114.896), swapRB=False)

    # Gender Prediction
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_BUCKETS[gender_preds[0].argmax()]

    # Age Prediction
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_BUCKETS[age_preds[0].argmax()]

    return gender, age
