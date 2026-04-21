import streamlit as st
import cv2
import dlib
import numpy as np
import pandas as pd
import joblib
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ======================
# 1. LOAD MODEL & ASSET (DENGAN CACHE)
# ======================
@st.cache_resource
def load_all_assets():
    try:
        # Gunakan path "Model" sesuai struktur foldermu
        scaler = joblib.load("Model/scaler_trained.pkl")
        model = joblib.load("Model/best_model_trained.pkl")
        feature_names = joblib.load("Model/feature_names.pkl")
        predictor = dlib.shape_predictor("Model/shape_predictor_68_face_landmarks.dat")
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        derived_features = [f for f in feature_names if f.endswith("_y")]
        emotion_labels = model.classes_
        
        return scaler, model, feature_names, predictor, face_cascade, derived_features, emotion_labels
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

assets = load_all_assets()
if assets:
    scaler, model, feature_names, predictor, face_cascade, derived_features, emotion_labels = assets

alpha = 0.2  # smoothing factor
st.title("😊 Real-Time Facial Emotion Recognition")
st.info("Tips: Pastikan wajah terkena cahaya yang cukup dan coba lepas kacamata jika deteksi sulit.")

# ======================
# 2. LANDMARK UTILS
# ======================
def align_and_crop_face(image, shape, desired_size=350, padding=0.35):
    shape = np.array(shape)
    left_eye = np.mean(shape[36:42], axis=0)
    right_eye = np.mean(shape[42:48], axis=0)
    dY, dX = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
    ones = np.ones((shape.shape[0], 1))
    shape_hom = np.hstack([shape, ones])
    rotated_landmarks = (M @ shape_hom.T).T
    x, y, w, h = cv2.boundingRect(rotated_landmarks.astype(np.int32))
    x_pad, y_pad = int(w * padding), int(h * padding)
    x1, y1 = max(0, x - x_pad), max(0, y - y_pad)
    x2, y2 = min(rotated.shape[1], x + w + x_pad), min(rotated.shape[0], y + h + y_pad)
    crop = rotated[y1:y2, x1:x2]
    if crop.size == 0: return None, None
    resized = cv2.resize(crop, (desired_size, desired_size))
    lm_crop = rotated_landmarks - np.array([x1, y1])
    scale_x, scale_y = desired_size / (x2 - x1), desired_size / (y2 - y1)
    lm_resized = np.zeros_like(lm_crop)
    lm_resized[:, 0], lm_resized[:, 1] = lm_crop[:, 0] * scale_x, lm_crop[:, 1] * scale_y
    return resized, lm_resized

def extract_landmarks(landmarks):
    data = {}
    for i, (x, y) in enumerate(landmarks):
        data[f"x{i}"] = x
        data[f"y{i}"] = y
    return data

# ======================
# 3. INFERENCE FUNCTION
# ======================
def predict_emotion(frame, prev_probs):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None, prev_probs

    for (x, y, w, h) in faces:
        # Gambarkan kotak merah di wajah untuk debug visual
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape_pts = [(p.x, p.y) for p in shape.parts()]
        
        _, aligned_landmarks = align_and_crop_face(frame, shape_pts)
        if aligned_landmarks is None: continue

        feat_dict = extract_landmarks(aligned_landmarks)
        for df in derived_features:
            feat_dict[df] = feat_dict.get(df.replace("_y", ""), 0)

        feat_vector = [feat_dict.get(col, 0) for col in feature_names]
        feat_df = pd.DataFrame([feat_vector], columns=feature_names)
        feat_scaled = scaler.transform(feat_df)

        if hasattr(model, "predict_proba"):
            raw_probs = model.predict_proba(feat_scaled)[0]
        else:
            pred_label = model.predict(feat_scaled)[0]
            raw_probs = np.zeros(len(emotion_labels))
            raw_probs[list(emotion_labels).index(pred_label)] = 1.0

        smooth_probs = raw_probs if prev_probs is None else (1 - alpha) * prev_probs + alpha * raw_probs
        return emotion_labels[int(np.argmax(smooth_probs))], smooth_probs
        
    return None, prev_probs

# ======================
# 4. WEBRTC CLASS (OPTIMIZED & DEBUG)
# ======================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.prev_probs = None
        self.frame_count = 0
        self.last_pred = "Mencari wajah..."

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Proses AI setiap 3 frame agar tidak terlalu berat
        if self.frame_count % 3 == 0:
            # Kirim gambar ke fungsi prediksi
            pred, self.prev_probs = predict_emotion(img, self.prev_probs)
            if pred:
                self.last_pred = pred
            else:
                self.last_pred = "Wajah tidak terdeteksi"

        # Tampilkan hasil di layar
        cv2.putText(img, f"Emotion: {self.last_pred}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img

# ======================
# 5. RUN STREAMER
# ======================
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="fer-realtime",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)