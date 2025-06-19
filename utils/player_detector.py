import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import defaultdict

def player_detector(frames):
    weights_path = "player_tracker/runs/detect/train2/weights/best.pt"  
    model = YOLO(weights_path)   
    results = model.track(frames, persist=True)
    annotated_frames = []
    player_positions = {
        1: [],  # Bottom half → Player 1
        2: []   # Top half → Player 2
    }

    frame_height = frames[0].shape[0]
    for result in results:
        frame = result.orig_img
        boxes = result.boxes.xywh
        ids = result.boxes.id
        if ids is None:
            annotated_frames.append(frame)
            continue
        for box in boxes:
            x_center, y_center, width, height = box
            if y_center >= frame_height / 2:
                player_positions[1].append((float(x_center), float(y_center)))  # bottom → player 1
            else:
                player_positions[2].append((float(x_center), float(y_center)))  # top → player 2
            x1, y1 = int(x_center - width / 2), int(y_center - height / 2)
            x2, y2 = int(x_center + width / 2), int(y_center + height / 2)
            color = (0, 255, 0) if y_center >= frame_height / 2 else (255, 0, 0)
            label = "P1" if y_center >= frame_height / 2 else "P2"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        annotated_frames.append(frame)
    return annotated_frames, player_positions

