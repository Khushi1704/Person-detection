import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
from sort.sort import Sort
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO('yolov8n.pt').to(device)
tracker = Sort()

video_path = 'C:\\Users\\ksute\\humantr\\output\\ABA Therapy_ Daniel - Communication.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

output_path = 'C:\\Users\\ksute\\humantr\\output\\processed_output.mp4'
if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def process_frame(frame, model, tracker):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 640))
        results = model(frame_resized)
        if not results:
            return frame
        detections = results[0].boxes
        if detections is None or detections.xyxy is None:
            return frame
        bbox_xyxy = detections.xyxy.cpu().numpy()
        class_ids = detections.cls.cpu().numpy()
        bbox_xywh = []
        for i, box in enumerate(bbox_xyxy):
            if class_ids[i] == 0:
                x1, y1, x2, y2 = box
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                bbox_xywh.append([x_center, y_center, w, h])
        if not bbox_xywh:
            return frame
        trackers = tracker.update(np.array(bbox_xywh))
        for track in trackers:
            x_center, y_center, w, h, track_id = track
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)
            label = 'Child' if h < 150 else 'Adult'
            color = (0, 255, 0) if label == 'Child' else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID: {int(track_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
    except Exception as e:
        print(f"An error occurred: {e}")
        return frame

while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("Empty or invalid frame encountered.")
        break
    frame_processed = process_frame(frame, model, tracker)
    if frame_processed is None:
        continue
    out.write(frame_processed)
    cv2.imshow('Processed Frame', frame_processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to {output_path}")
