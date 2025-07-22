"""
This code reads a video file and an accompanying text file containing bounding box coordinates.
It overlays the bounding boxes onto the corresponding video frames and saves a new annotated video.
The output video is saved with an FPS of 1 as required.

@ IntentVC Challenge, Date: 2025-03-12
"""

import cv2
import numpy as np
import os
from glob import glob
import shutil
from tqdm import tqdm
from collections import defaultdict

# for video_path in glob('/data/mm2025/IntentVC/*/*/*.mp4'):

#     # Open video
#     cap = cv2.VideoCapture(video_path)

#     # Get video properties
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     print(video_path.split('/')[-1], frame_width, frame_height)
# exit()

def normalize_coordinates(box, image_width, image_height):
    x1, y1, x2, y2 = box
    normalized_box = [
        min(max(round((x1 / image_width) * 1000), 0), 1000),
        min(max(round((y1 / image_height) * 1000), 0), 1000),
        max(min(round((x2 / image_width) * 1000), 1000), 0),
        max(min(round((y2 / image_height) * 1000), 1000), 0)
    ]
    return normalized_box

def return45(x):
    xl = int(x)
    return xl if x-xl < 0.5 else xl+1

# Define paths
data_source_folder = glob(f"/data/mm2025/IntentVC/*/*/*.mp4")
for video_path in tqdm(data_source_folder):
    obj = video_path.split('/')[-1].split('.')[0]
    bboxes_path = video_path.replace(video_path.split('/')[-1], 'object_bboxes.txt')
    


    # Read bounding boxes
    with open(bboxes_path, 'r') as f:
        bboxes = [list(map(int, line.strip().split(','))) for line in f]

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Get video properties-
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    normalized_bboxes = []
    for bbox in bboxes:
        x, y, w, h = bbox
        normalized_bboxes.append(normalize_coordinates((x, y, x+w, y+h), image_width=frame_width, image_height=frame_height))
    normalized_bboxes = np.asarray(normalized_bboxes)
    np.save(bboxes_path.replace('object_bboxes.txt', 'object_bboxes.npy'), normalized_bboxes)

    cap.release()

print(f"Annotated video saved.")
