import streamlit as st
import cv2
import pickle
import cvzone
import numpy as np
import os
import time

st.set_page_config(page_title="Parking Space Detection", layout="wide")
st.title("🚗 Real-Time Parking Space Detection")

st.markdown("""
This web application processes a local video feed and detects available parking spaces using OpenCV.
Upload or configure your space positions using the desktop picker script, and preview the results here!
""")

# Load positions
pos_file = 'carParkPos' if os.path.exists('carParkPos') else 'CarParkPos'
try:
    with open(pos_file, 'rb') as f:
        posList = pickle.load(f)
except FileNotFoundError:
    st.error(f"Could not find the coordinates file: {pos_file}. Please create one using the desktop picker script.")
    posList = []

width, height = 107, 48

def checkParkingSpace(imgPro, img):
    spaceCounter = 0
    for pos in posList:
        x, y = pos
        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        if count < 900:
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1,
                           thickness=2, offset=0, colorR=color)

    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                           thickness=5, offset=20, colorR=(0, 200, 0))

col1, col2 = st.columns([1, 4])

with col1:
    run_video = st.checkbox("🟢 Run Video Stream", value=True)
    st.info("Check the box to start processing the video. Uncheck it to stop.")

# Cache the video locally on Streamlit server memory so it isn't deleted on rerun
@st.cache_resource
def get_cap():
    return cv2.VideoCapture('carPark.mp4')

cap = get_cap()

with col2:
    frame_placeholder = st.empty()
    
    if run_video:
        if not cap.isOpened():
            st.error("Error: Could not process carPark.mp4. Ensure file exists and codecs are available.")
        else:
            # Revert to a while-loop but use an empty placeholder to update DOM in-place seamlessly without flashing
            while run_video:
                if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    
                success, img = cap.read()
                if success:
                    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
                    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
                    imgMedian = cv2.medianBlur(imgThreshold, 5)
                    kernel = np.ones((3, 3), np.uint8)
                    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

                    checkParkingSpace(imgDilate, img)
                    
                    # Resize the frame down to fit Streamlit websocket payloads without crashing
                    imgDisplay = cv2.resize(img, (800, 450))
                    
                    # Convert to JPG bytes with 60% quality compression!
                    ret, buffer = cv2.imencode('.jpg', imgDisplay, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    if ret:
                        # Updates the tag silently, preventing Streamlit clearing UI and flashing black
                        frame_placeholder.image(buffer.tobytes(), use_column_width=True)
                        
                    # Target 12 FPS to fit perfectly under WebSocket traffic limits
                    time.sleep(0.08)
                else:
                    st.warning("Video stream ended or failed to read.")
                    break
    else:
        st.info("Video feed is currently stopped.")
