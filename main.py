import cv2
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
from utils import ball_detector, player_detector, CourtLineDetector
import numpy as np

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

def read_video(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open the input video: {video_path}")
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return frames

def main():
    input_video_path = "/Users/lokesh/Desktop/cv_project/input_video.mp4"
    video_frames = read_video(input_video_path)
    court_line_detector = CourtLineDetector()
    court_keypoints = court_line_detector.predict(video_frames[0])
    output_video_frames  = court_line_detector.draw_keypoints_on_video(video_frames, court_keypoints)
    os.makedirs("output_videos", exist_ok=True)
    save_video(output_video_frames, "output_videos/output_video6.avi")


if __name__ == "__main__":
    main()





