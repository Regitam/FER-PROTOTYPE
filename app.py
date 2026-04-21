import streamlit as st
import cv2
import dlib
import numpy as np
import pandas as pd
import joblib
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ======================
# LOAD MODEL & ASSET
# ======================
scaler = joblib.load("Model/scaler_trained.pkl")
model = joblib.load("Model/best_model_trained.pkl")
feature_names = joblib.load("Model/feature_names.pkl")
predictor = dlib.shape_predictor("Model/shape_predictor_68_face_landmarks.dat")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

derived_features = [f for f in feature_names if f.endswith("_y")]
emotion_labels = model.classes_

alpha = 0.2

st.title("Real-Time Facial Emotion Recognition")

# ======================
# LANDMARK UTIL
# ======================
def align_and_crop_face(image, shape, desired_size=350, padding=0.35):
    shape = np.array(shape)

    left_eye = np.mean(shape[36:42], axis=0)
    right_eye = np.mean(shape[42:48], axis=0)

    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                   (left_eye[1] + right_eye[1]) / 2)

    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

    rotated = cv2.warpAffine(image, M,
                             (image.shape[1], image.shape[0]),
                             flags=cv2.INTER_CUBIC)

    ones = np.ones((shape.shape[0], 1))
    shape_hom = np.hstack([shape, ones])
    rotated_landmarks = (M @ shape_hom.T).T

    x, y, w, h = cv2.boundingRect(rotated_landmarks.astype(np.int32))

    x_pad, y_pad = int(w * padding), int(h * padding)

    x1, y1 = max(0, x - x_pad), max(0, y - y_pad)
    x2, y2 = min(rotated.shape[1], x + w + x_pad), min(rotated.shape[0], y + h + y_pad)

    crop = rotated[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    resized = cv2.resize(crop, (desired_size, desired_size))

    lm_crop = rotated_landmarks - np.array([x1, y1])

    scale_x = desired_size / (x2 - x1)
    scale_y = desired_size / (y2 - y1)

    lm_resized = np.zeros_like(lm_crop)
    lm_resized[:, 0] = lm_crop[:, 0] * scale_x
    lm_resized[:, 1] = lm_crop[:, 1] * scale_y

    return resized, lm_resized


def extract_all_landmarks(landmarks):
    data = {}
    for i, (x, y) in enumerate(landmarks):
        data[f"x{i}"] = x
        data[f"y{i}"] = y
    return data


# ======================
# INFERENCE PIPELINE
# ======================
def predict_emotion(frame, prev_probs):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray)

    faces = face_cascade.detectMultiScale(gray, 1.1, 3)

    if len(faces) == 0:
        return None, prev_probs

    for (x, y, w, h) in faces:
        rect = dlib.rectangle(x, y, x + w, y + h)

        try:
            shape = predictor(gray, rect)
            shape_pts = [(p.x, p.y) for p in shape.parts()]

            aligned_face, aligned_landmarks = align_and_crop_face(frame, shape_pts)
            if aligned_landmarks is None:
                continue

            feat_dict = extract_all_landmarks(aligned_landmarks)

            for df in derived_features:
                base = df.replace("_y", "")
                feat_dict[df] = feat_dict.get(base, 0)

            feat_vector = [feat_dict.get(col, 0) for col in feature_names]
            feat_df = pd.DataFrame([feat_vector], columns=feature_names)

            feat_scaled = scaler.transform(feat_df)

            if hasattr(model, "predict_proba"):
                raw_probs = model.predict_proba(feat_scaled)[0]
            else:
                pred_label = model.predict(feat_scaled)[0]
                raw_probs = np.zeros(len(emotion_labels))
                raw_probs[list(emotion_labels).index(pred_label)] = 1.0

            # smoothing
            if prev_probs is None:
                smooth_probs = raw_probs
            else:
                smooth_probs = (1 - alpha) * prev_probs + alpha * raw_probs

            pred_idx = np.argmax(smooth_probs)
            pred = emotion_labels[pred_idx]

            return pred, smooth_probs

        except Exception as e:
            print("Error:", e)
            continue

    return None, prev_probs


# ======================
# WEBRTC PROCESSOR
# ======================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.prev_probs = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 3)

        # DEBUG TEXT
        cv2.putText(img, "STREAM ACTIVE",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255), 1)

        cv2.putText(img, f"faces: {len(faces)}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255), 1)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            try:
                rect = dlib.rectangle(x, y, x + w, y + h)
                shape = predictor(gray, rect)
                shape_pts = [(p.x, p.y) for p in shape.parts()]

                # LANDMARK
                for (lx, ly) in shape_pts:
                    cv2.circle(img, (lx, ly), 1, (0, 255, 0), -1)

                pred, self.prev_probs = predict_emotion(img, self.prev_probs)

                if pred is not None:
                    cv2.putText(img,
                                f"Emotion: {pred}",
                                (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2)

            except Exception as e:
                print("ERROR:", e)

        return img


# ======================
# STREAMLIT RUN
# ======================
webrtc_streamer(
    key="fer-realtime",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)