import streamlit as st
import cv2
import pickle
import cvzone
import numpy as np
import os
import subprocess

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

# By using st.cache_resource, this entire intensive AI compute runs exactly once when the server boots!
@st.cache_resource(show_spinner=False)
def process_full_video(source_path, target_path):
    if os.path.exists(target_path):
        return True

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        return False
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_path = "temp_out.mp4"
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width_vid, height_vid))
    
    progress_bar = st.progress(0, text="AI is computing parking space layout over video...")
    
    frame_count = 0
    while True:
        success, img = cap.read()
        if not success:
            break
            
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
        imgMedian = cv2.medianBlur(imgThreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

        checkParkingSpace(imgDilate, img)
        
        out.write(img)
        frame_count += 1
        
        # Throttled progress update to keep UI fast
        if frame_count % 15 == 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0), text=f"Processing AI Frame {frame_count}/{total_frames}...")
            
    cap.release()
    out.release()
    
    progress_bar.progress(1.0, text="Encoding optimized playback video for the Web (H.264)...")
    
    # Convert OpenCV raw mp4 to HTML5 compatible H.264 mp4 via ffmpeg
    subprocess.run(['ffmpeg', '-y', '-i', temp_path, '-vcodec', 'libx264', '-f', 'mp4', target_path])
    os.remove(temp_path)
    
    # Clear the progress bar when perfectly complete
    progress_bar.empty()
    return True

col1, col2 = st.columns([1, 4])

with col1:
    st.info("✅ High-Speed WebSocket free architecture.\n\n"
            "The AI executes the layout detection matrix exactly once locally, converting it into a perfectly "
            "compressed H.264 stream for native zero-lag browser playback!")

with col2:
    if len(posList) > 0:
        with st.spinner("Initializing AI Computer Vision Engine..."):
            success = process_full_video('carPark.mp4', 'output.mp4')
            
        if success and os.path.exists('output.mp4'):
            # The holy grail of video streaming in Streamlit: the native HTML5 player
            st.video('output.mp4', loop=True, autoplay=True, muted=True)
        else:
            st.error("Failed to generate AI processed video playback.")
