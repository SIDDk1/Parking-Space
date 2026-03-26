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
    run_video = st.button("Start Video Feed")
    stop_video = st.button("Stop Video Feed")
    st.info("The video feed uses a loop. Click 'Stop Video Feed' or refresh the page to stop.")

with col2:
    frame_placeholder = st.empty()

if run_video:
    cap = cv2.VideoCapture('carPark.mp4')
    if not cap.isOpened():
        st.error("Error opening video file `carPark.mp4`")
    else:
        while True:
            # Check if stop button pressed internally? No, Streamlit buttons rerun the script.
            # But the while loop will block until stopped. 
            # In Streamlit, a long running loop blocks the session. A quick way to stop it natively is to use session state or just tell users to refresh.
            if stop_video:
                break
            
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            success, img = cap.read()
            if not success:
                st.warning("Video stream ended or failed to read.")
                break
                
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
            imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
            imgMedian = cv2.medianBlur(imgThreshold, 5)
            kernel = np.ones((3, 3), np.uint8)
            imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

            checkParkingSpace(imgDilate, img)
            
            # Convert BGR to RGB for Streamlit image rendering
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(imgRGB, channels="RGB", use_container_width=True)
            
            # Sleep to prevent overwhelming the Streamlit Cloud websocket (20 FPS max)
            time.sleep(0.05)
