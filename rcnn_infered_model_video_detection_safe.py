# rcnn_infered_model_video_detection_safe.py

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import argparse
from pathlib import Path
import json 
from PIL import Image, ImageDraw

from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import cv2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup
class_names = ["__background__", "objects", "bag", "bra", "clothe", "shoe"]
num_classes = len(class_names)
threshold = 0.5
resize_width = 640 

# Load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, hidden_layer, num_classes)

model.load_state_dict(torch.load("maskrcnn_sam2.pth", map_location=device))
model.to(device)
model.eval()

# Open video
cap = cv2.VideoCapture("D:\\Cours\\UniversityWest\\masterThesis\\camera2_record_2025-01-27_partial\\camera2_record_2025-01-27 14-23-08_is_slower.avi")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    scale = resize_width / width
    frame = cv2.resize(frame, (resize_width, int(height * scale)))

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            prediction = model(img_tensor)[0]
            torch.cuda.synchronize()
    except RuntimeError as e:
        logging.error(f"CUDA error during model inference: {e}")
        break

    boxes = prediction.get('boxes', [])
    scores = prediction.get('scores', [])
    labels = prediction.get('labels', [])
    masks = prediction.get('masks', None)

    num_preds = len(boxes)
    num_masks = masks.shape[0] if masks is not None else 0

    for i in range(num_preds):
        score = scores[i].item()
        if score < threshold:
            continue

        if i >= num_masks:
            logging.warning(f"Skipping mask index {i}: out of bounds (total masks: {num_masks})")
            continue

        box = boxes[i].detach().cpu().numpy().astype(int)
        label_id = labels[i].item()
        label_name = class_names[label_id] if label_id < len(class_names) else f"unknown_{label_id}"

        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label_name}: {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        try:
            mask = masks[i, 0].detach().cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8) * 255
            colored_mask = np.stack([np.zeros_like(mask), mask, np.zeros_like(mask)], axis=2)
            frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)
        except Exception as e:
            logging.error(f"Failed to apply mask at index {i}: {e}")

    cv2.imshow("Mask R-CNN Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
