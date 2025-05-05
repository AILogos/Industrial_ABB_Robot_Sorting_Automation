
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

class CocoSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_path, transforms=None):
        self.img_dir = Path(img_dir)
        self.transforms = transforms

        with open(ann_path) as f:
            coco = json.load(f)

        self.images = coco['images']
        self.annotations = coco['annotations']
        self.categories = coco['categories']

        # Build index: image_id -> annotations
        self.image_id_to_anns = {}
        for ann in self.annotations:
            self.image_id_to_anns.setdefault(ann['image_id'], []).append(ann)

        self.category_map = {cat['id']: i+1 for i, cat in enumerate(self.categories)}

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image_id = img_info['id']
        img_path = self.img_dir / img_info['file_name']

        img = Image.open(img_path).convert("RGB")
        anns = self.image_id_to_anns.get(image_id, [])

        boxes, labels, masks = [], [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(self.category_map[ann['category_id']])

            # Segmentation polygons -> mask
            mask = Image.new('L', (img_info['width'], img_info['height']))
            for seg in ann['segmentation']:
                seg = np.array(seg).reshape(-1, 2)
                #Image.Draw.Draw(mask).polygon([tuple(p) for p in seg], outline=1, fill=1)
                draw = ImageDraw.Draw(mask)
                draw.polygon([tuple(p) for p in seg], outline=1, fill=1)

            masks.append(np.array(mask))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([image_id])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)

# CLI
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="ClothesObjectsSegmentation.v1-quick_test.coco-segmentation/valid", help='Image directory')
parser.add_argument('--ann', type=str, default="ClothesObjectsSegmentation.v1-quick_test.coco-segmentation/valid/_annotations.coco.json", help='COCO JSON path')
parser.add_argument('--model', type=str, default="maskrcnn_sam2.pth", help='Trained model path')
parser.add_argument('--thresh', type=float, default=0.5, help='Confidence threshold')
parser.add_argument('--outdir', type=str, default="predictions", help='Save output directory')
args = parser.parse_args()

# Setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 6
class_names = ["__background__", "objects", "bag", "bra", "clothe", "shoe"]

# Load model
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, hidden_layer, num_classes)

model.load_state_dict(torch.load(args.model, map_location=device))
model.to(device)
model.eval()

# Load val set
dataset = CocoSegmentationDataset(img_dir=args.data, ann_path=args.ann, transforms=T.ToTensor())
os.makedirs(args.outdir, exist_ok=True)

# Run prediction loop
for i in range(min(5, len(dataset))):
    img, target = dataset[i]
    img_vis = img.permute(1, 2, 0).numpy()
    img_input = img.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_input)[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Ground Truth
    ax1.imshow(img_vis)
    for mask in target['masks']:
        m = np.ma.masked_where(mask.numpy() < 0.5, mask.numpy())
        ax1.imshow(m, alpha=0.4, cmap='Greens')
    ax1.set_title("Ground Truth")
    ax1.axis('off')

    # Right: Prediction
    ax2.imshow(img_vis)
    for j in range(len(prediction['boxes'])):
        score = prediction['scores'][j].item()
        if score < args.thresh:
            continue
        mask = prediction['masks'][j, 0].cpu().numpy()
        mask = np.ma.masked_where(mask < 0.5, mask)
        ax2.imshow(mask, cmap='cool', alpha=0.4)

        box = prediction['boxes'][j].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        ax2.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    edgecolor='lime', facecolor='none', linewidth=2))
        label_id = prediction['labels'][j].item()
        label_name = class_names[label_id] if label_id < len(class_names) else str(label_id)
        ax2.text(x1, y1 - 5, f"{label_name} ({score:.2f})",
                 color='lime', fontsize=12, backgroundcolor='black')
    ax2.set_title("Predicted")
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"compare_{i+1}.png"), bbox_inches='tight')
    plt.close()
