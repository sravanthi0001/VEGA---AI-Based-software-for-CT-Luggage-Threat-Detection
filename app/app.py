import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import datetime
from ultralytics import YOLO
from PIL import Image

# ------------------- CONFIG -------------------
MODEL_PATH = 'runs/train/yolov8n_custom/weights/best.pt'  # Update if your best.pt is elsewhere
LOG_PATH = 'app/detection_logs.csv'
THREAT_CLASSES = ["knife", "gun", "explosive"]

# ------------------- UTILS -------------------
def load_model(model_path=MODEL_PATH):
    """Load YOLOv8 model from file."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def detect_image(model, image):
    """Run YOLOv8 detection on a PIL image. Returns annotated image, results list."""
    img_array = np.array(image.convert('RGB'))
    results = model(img_array)
    boxes = results[0].boxes
    annotated_img = results[0].plot()
    detected = []
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls]
        detected.append({"name": name, "confidence": conf})
    return annotated_img, detected

def detect_video(model):
    """Run YOLOv8 detection on webcam stream. Yields annotated frames and detection info."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access webcam.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        boxes = results[0].boxes
        annotated_frame = results[0].plot()
        detected = []
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]
            detected.append({"name": name, "confidence": conf})
        yield annotated_frame, detected
    cap.release()

def threat_analysis(detected):
    """Check if any detected object is a threat."""
    threats = [d for d in detected if d["name"].lower() in THREAT_CLASSES]
    return threats

def log_detection(detected):
    """Log detection results to CSV."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for d in detected:
        rows.append({
            "timestamp": timestamp,
            "object": d["name"],
            "confidence": round(d["confidence"], 3)
        })
    df = pd.DataFrame(rows)
    if os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_PATH, mode='w', header=True, index=False)

# ------------------- UI -------------------
st.set_page_config(page_title="Airport Luggage Threat Detection System", layout="wide")
st.title("🛄 Airport Luggage Threat Detection System")

menu = st.sidebar.radio("Select Mode", ("Upload CT Scan Image", "Live Camera Detection"))

model = load_model()

if menu == "Upload CT Scan Image":
    st.subheader("Upload a CT Scan Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file and model:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Detect Threats"):
            annotated_img, detected = detect_image(model, image)
            st.image(annotated_img, caption="Detection Result", use_column_width=True)
            if detected:
                st.write("### Detected Objects:")
                for d in detected:
                    st.write(f"- {d['name']} (Confidence: {d['confidence']:.2f})")
                threats = threat_analysis(detected)
                if threats:
                    st.error("🚨 Threat Detected!")
                log_detection(detected)
            else:
                st.success("No objects detected.")

elif menu == "Live Camera Detection":
    st.subheader("Live Camera Threat Detection")
    if model:
        run = st.button("Start Camera")
        stop = st.button("Stop Camera")
        if run:
            frame_placeholder = st.empty()
            info_placeholder = st.empty()
            try:
                for frame, detected in detect_video(model):
                    frame_placeholder.image(frame, channels="BGR")
                    if detected:
                        info_placeholder.write(
                            ", ".join([f"{d['name']} ({d['confidence']:.2f})" for d in detected])
                        )
                        threats = threat_analysis(detected)
                        if threats:
                            st.error("🚨 Threat Detected!")
                        log_detection(detected)
                    else:
                        info_placeholder.write("No objects detected.")
            except Exception as e:
                st.error(f"Camera error: {e}")
