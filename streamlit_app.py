import streamlit as st
import tempfile
import cv2
import os
import numpy as np
from utils import  player_detector, CourtLineDetector

def compute_homography():
        image_pts = np.array([
        [590, 310],    # Top-left
        [1352, 310],   # Top-right
        [338, 858],    # Bottom-left
        [1581, 857]    # Bottom-right
        ], dtype=np.float32)

        world_pts = np.array([
            [0, 0],         # Top-left
            [8.23, 0],      # Top-right
            [0, 23.77],     # Bottom-left
            [8.23, 23.77]   # Bottom-right
        ], dtype=np.float32)
        H, _ = cv2.findHomography(image_pts, world_pts)
        return H

def pixel_to_world(x, y, H):
    pt = np.array([x, y, 1]).reshape(3, 1)
    world = H @ pt
    world /= world[2]
    return float(world[0]), float(world[1])

def compute_speeds(positions, fps):
    speeds = []
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        dist = (dx ** 2 + dy ** 2) ** 0.5
        speeds.append(dist * fps)
    return speeds, np.mean(speeds) if speeds else 0

def compute_total_distance(positions):
    return sum(
        ((positions[i][0] - positions[i - 1][0]) ** 2 +
        (positions[i][1] - positions[i - 1][1]) ** 2) ** 0.5
        for i in range(1, len(positions))
    )

def compute_side_percentages(positions, reverse=False):
    left_count = 0
    for x, y in positions:
        if reverse:
            if x >= 4.115:
                left_count += 1
        else:
            if x < 4.115:
                left_count += 1
    total = len(positions)
    left_pct = (left_count / total) * 100 if total else 0
    right_pct = 100 - left_pct
    return left_pct, right_pct

st.set_page_config(layout="wide")
st.title("Tennis Player Performance Analyzer")
uploaded_file = st.file_uploader("Upload a match video", type=["mp4", "avi", "mov"])
if uploaded_file:
    st.video(uploaded_file)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    st.info("Analyzing player movement...")
    annotated_frames, player_positions_px = player_detector(frames)
    H = compute_homography()
    fps = 24
    player_positions_m = {
        pid: [pixel_to_world(x, y, H) for (x, y) in pos_list]
        for pid, pos_list in player_positions_px.items()
    }
    for pid in sorted(player_positions_m.keys()):
        speeds, avg_speed = compute_speeds(player_positions_m[pid], fps)
        total_distance = compute_total_distance(player_positions_m[pid])
        is_player2 = (pid == 2)
        left_pct, right_pct = compute_side_percentages(player_positions_m[pid], reverse=is_player2)
        st.subheader(f"Player {pid} Stats")
        st.markdown(f"**Average Speed:**{avg_speed:.2f} m/s")
        st.markdown(f"**Total Distance:**{total_distance:.2f} meters")
        st.markdown(f"**Left Side Time:**{left_pct:.2f}%")
        st.markdown(f"**Right Side Time:**{right_pct:.2f}%")
    st.success("Analysis complete")
