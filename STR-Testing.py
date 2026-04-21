import streamlit as st
import cv2
import dlib
import numpy as np
import pandas as pd
import joblib
import time
from PIL import Image
import altair as alt

# ======================
# 1. KONFIGURASI HALAMAN
# ======================
st.set_page_config(page_title="Facial Emotion Recognition Prototype ",  layout="wide")

# ======================
# 2. LOAD ASSETS (CACHED)
# ======================
@st.cache_resource
def load_all_assets():
    try:
        # PENTING: Sesuaikan path folder 'Model' (M Kapital)
        scaler = joblib.load("Model/scaler_trained.pkl")
        model = joblib.load("Model/best_model_trained.pkl")
        feature_names = joblib.load("Model/feature_names.pkl")
        predictor = dlib.shape_predictor("Model/shape_predictor_68_face_landmarks.dat")
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        derived_features = [f for f in feature_names if f.endswith("_y")]
        emotion_labels = model.classes_
        return scaler, model, feature_names, predictor, face_cascade, derived_features, emotion_labels
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None

assets = load_all_assets()
if assets:
    scaler, model, feature_names, predictor, face_cascade, derived_features, emotion_labels = assets

# ======================
# 3. PREPROCESSING UTILS
# ======================
def align_and_crop_face(image, shape, desired_size=350, padding=0.35):
    shape = np.array(shape)
    left_eye = np.mean(shape[36:42], axis=0)
    right_eye = np.mean(shape[42:48], axis=0)

    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
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

def extract_all_landmarks(landmarks):
    data = {}
    for i, (lx, ly) in enumerate(landmarks):
        data[f"x{i}"] = lx
        data[f"y{i}"] = ly
    return data

# ======================
# 4. CORE PROCESSING FUNCTION
# ======================
def process_single_image(image_rgb):
    # Buat salinan untuk digambar
    img_draw = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_draw, cv2.COLOR_BGR2GRAY)
    start_time = time.time()
    
    # Deteksi Wajah
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return image_rgb, None, 0, None

    best_emotion = "Tidak Terdeteksi"
    probs_dict = {}

    for (x, y, w, h) in faces:
        # Gambar kotak visual (HIJAU)
        cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        
        # Gambar 68 titik: HIJAU, LEBIH BOLD (radius 2)
        for i in range(68):
            p = shape.part(i)
            # Warna Hijau, ukuran sedikit lebih besar (radius 2) agar Bold
            cv2.circle(img_draw, (p.x, p.y), 2, (0, 255, 0), -1)

        # Preprocessing & Inference
        shape_pts = [(p.x, p.y) for p in shape.parts()]
        # Gunakan image_rgb asli untuk preprocessing agar tidak ada coretan kotak/titik
        img_for_proc = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        _, aligned_landmarks = align_and_crop_face(img_for_proc, shape_pts)
        
        if aligned_landmarks is not None:
            feat_dict = extract_all_landmarks(aligned_landmarks)
            for df in derived_features:
                feat_dict[df] = feat_dict.get(df.replace("_y", ""), 0)

            feat_vector = [feat_dict.get(col, 0) for col in feature_names]
            feat_df = pd.DataFrame([feat_vector], columns=feature_names)
            feat_scaled = scaler.transform(feat_df)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(feat_scaled)[0]
                probs_dict = {label: float(prob) for label, prob in zip(emotion_labels, probs)}
                best_emotion = emotion_labels[np.argmax(probs)]
            else:
                best_emotion = str(model.predict(feat_scaled)[0])
                probs_dict = {best_emotion: 1.0}

            # TULIS EMOSI DALAM GAMBAR
            if best_emotion:
                label = best_emotion.upper()
                text_x, text_y = x, y + h + 30 
                
                # Background hitam untuk teks
                (t_w, t_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(img_draw, (text_x - 5, text_y - t_h - 10), (text_x + t_w + 5, text_y + 10), (0,0,0), -1)
                
                # Teks warna Hijau
                cv2.putText(img_draw, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    process_time = time.time() - start_time
    img_final = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    return img_final, best_emotion, process_time, probs_dict

# ======================
# 5. UI LAYOUT
# ======================
st.header("FER Prototype")
st.write("Silakan ambil foto untuk dianalisis emosinya.")

col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("📸 1. Ambil Foto")
    camera_photo = st.camera_input("Klik 'Take Photo'...")

if camera_photo is not None:
    img = Image.open(camera_photo)
    img_rgb = np.array(img)

    with col_output:
        st.subheader("🔍 2. Hasil Analisis")
        with st.spinner('Dlib sedang menganalisis wajah...'):
            final_img, prediction, exec_time, probabilities = process_single_image(img_rgb)
            
            # Tampilkan gambar hasil deteksi
            st.image(final_img, caption=f"Processing Time: {exec_time:.3f}s", use_container_width=True)
            
            # Menampilkan Probabilitas Bar Chart
            if prediction and probabilities:
                st.write("---")
                st.write("**Grafik Probabilitas Emosi:**")
                chart_df = pd.DataFrame.from_dict(probabilities, orient='index', columns=['Prob']).reset_index()
                chart_df = chart_df.rename(columns={'index': 'Emosi'})
                
                chart = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X('Emosi', sort='-y', title='Kategori Emosi'),
                    y=alt.Y('Prob', title='Keyakinan AI (0-1)'),
                    color=alt.Color('Emosi', legend=None)
                ).properties(height=350)
                
                # Perbaikan error: use_container_width=True (versi terbaru Streamlit)
                st.altair_chart(chart, use_container_width=True)
else:
    with col_output:
        st.info("Kamera aktif")

st.divider()
st.caption("Prototype FER © 2026 | Dlib Shape Predictor | Streamlit Web Hosting")