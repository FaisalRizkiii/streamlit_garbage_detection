import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- Konfigurasi ---
MODEL_PATH = 'best.pt'
st.set_page_config(layout="wide", page_title="Garbage Detection")
@st.cache_resource
def load_model():
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Fungsi Prediksi ---
def predict_and_plot(image_source):
    if model is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    results = model.predict(image_source, conf=0.5, verbose=False)

    annotated_frame_bgr = results[0].plot()

    annotated_frame_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)

    return annotated_frame_rgb

# --- Layout Utama ---
st.title("Aplikasi Deteksi Sampah")

# Sidebar Menu
menu = ["Image", "Video", "Real-time Webcam"]
choice = st.sidebar.selectbox("Pilih Mode", menu)

# ==================== MENU 1: IMAGE ====================
if choice == "Image":
    st.header("Deteksi Gambar")
    uploaded_file = st.file_uploader("Upload gambar...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Gambar Asli")
            st.image(image_pil, use_container_width=True)

        with col2:
            st.subheader("Hasil Prediksi")
            with st.spinner('Memproses...'):
                result_image_rgb = predict_and_plot(image_pil)
                st.image(result_image_rgb, use_container_width=True)

# ==================== MENU 2: VIDEO (LOOPING) ====================
elif choice == "Video":
    st.header("Deteksi Video")
    uploaded_video = st.file_uploader("Upload video...", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        stop_video_btn = st.button("Stop Video Processing")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Video Asli")
            original_frame_slot = st.empty() # Placeholder kiri
        with col2:
            st.subheader("Hasil Prediksi")
            processed_frame_slot = st.empty() # Placeholder kanan

        cap = cv2.VideoCapture(tfile.name)

        while cap.isOpened() and not stop_video_btn:
            ret, frame_bgr = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            original_frame_slot.image(frame_rgb, channels="RGB", use_container_width=True)

            result_frame_rgb = predict_and_plot(frame_bgr)
            processed_frame_slot.image(result_frame_rgb, channels="RGB", use_container_width=True)

        cap.release()

# ==================== MENU 3: REAL-TIME WEBCAM ====================
elif choice == "Real-time Webcam":
    st.header("Deteksi Video")
    stop_webcam_btn = st.button("Stop Webcam")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Feed Asli")
        webcam_raw_slot = st.empty()
    with col2:
        st.subheader("Feed Prediksi")
        webcam_processed_slot = st.empty()

    cap = cv2.VideoCapture(0)

    while cap.isOpened() and not stop_webcam_btn:
        ret, frame_bgr = cap.read()
        if not ret:
            st.error("Gagal menangkap gambar dari kamera.")
            break

        frame_rgb_raw = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        webcam_raw_slot.image(frame_rgb_raw, channels="RGB", use_container_width=True)
        result_frame_rgb = predict_and_plot(frame_bgr)
        webcam_processed_slot.image(result_frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
    st.write("Webcam dihentikan.")