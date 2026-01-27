import os
import json
import time
import torch
import argparse
from PIL import Image
from typing import List
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import DetrForObjectDetection
from eval.utils import load_config


torch.set_grad_enabled(False)

CLASSES = [  # COCO Classes
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


class ImageDataset(Dataset):
    def __init__(self, image_folder: str, image_paths: List[str]):
        self.image_folder = image_folder
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.image_folder, self.image_paths[idx])
        image = Image.open(path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor, self.image_paths[idx]


def load_image_list(json_path: str) -> List[str]:
    images = set()
    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
        except:
            data = [json.loads(line) for line in f]
    for entry in data:
        if "image" in entry:
            images.add(entry["image"])
    return list(images)


def detect(images: torch.Tensor, model, threshold: float):
    outputs = model(images)
    logits = outputs.logits.softmax(-1)[:, :, :-1]
    keep = logits.max(-1).values > threshold

    boxes = outputs.pred_boxes
    results = []
    for i in range(len(images)):
        probs = logits[i][keep[i]]
        boxes_scaled = boxes[i][keep[i]]
        results.append((probs, boxes_scaled))
    return results


def save_results(results: list, output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--th", type=float, default=0.95)
    parser.add_argument("--metric", type=str, default="pope")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    question_path, image_dir = load_config(args.metric, args.dataset)

    image_list = load_image_list(question_path)

    dataset = ImageDataset(image_dir, image_list)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model.eval().cuda()

    start = time.time()
    print(f"[INFO] Running detection with threshold {args.th}...")

    result_batch = []
    for image_tensors, image_names in dataloader:
        image_tensors = image_tensors.cuda()
        detections = detect(image_tensors, model, args.th)

        for name, (probs, boxes) in zip(image_names, detections):
            cl_ids = probs.argmax(dim=1)
            cl_names = [CLASSES[cl] for cl in cl_ids]
            cl_scores = [round(float(p[c]), 4) for p, c in zip(probs, cl_ids)]
            result_batch.append({"image": name, "objects": cl_names})

    save_results(result_batch, args.save_path)

    print(f"[INFO] Detection complete. Time taken: {round(time.time() - start, 2)}s")
    print(f"[INFO] Results saved to {args.save_path}")
