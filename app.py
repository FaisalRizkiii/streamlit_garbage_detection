import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- Konfigurasi ---
st.set_page_config(layout="wide", page_title="Garbage Detection")

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        return YOLO('best.pt')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Callback untuk memproses setiap frame ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    if model is not None:
        results = model.predict(img, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
    else:
        annotated_frame = img
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- Layout Utama ---
st.title("Real-Time Garbage Detection")

# Sidebar
menu = ["Image", "Video", "Live Webcam"]
choice = st.sidebar.selectbox("Pilih Mode", menu)

# ==================== MODE 1: IMAGE ====================
if choice == "Image":
    st.header("Upload Gambar")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", channels="RGB")
        
        if model:
            res = model.predict(img, conf=0.5)
            res_plotted = res[0].plot()
            st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="Result", channels="RGB")

# ==================== MODE 2: VIDEO ====================
elif choice == "Video":
    st.header("Upload Video")
    uploaded_video = st.file_uploader("", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if model:
                res = model.predict(frame, conf=0.5, verbose=False)
                res_plotted = res[0].plot()
                stframe.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), channels="RGB")
        cap.release()

# ==================== MODE 3: LIVE WEBCAM (Real-Time) ====================
elif choice == "Live Webcam":
    st.header("Live Detection")

    # Konfigurasi agar bisa jalan di Cloud (STUN Server)
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Menjalankan WebRTC Streamer
    webrtc_streamer(
        key="garbage-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "width": {"min": 640, "ideal": 640},
                "height": {"min": 640, "ideal": 640},
            },
            "audio": False,
        },
        async_processing=True,
    )