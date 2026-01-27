import argparse
import json
import os
from typing import List
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference, get_transform
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
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
    def __init__(self, image_folder, image_paths):
        self.image_folder = image_folder
        self.image_paths = image_paths
        self.transform = get_transform()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        full_path = os.path.join(self.image_folder, image_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(full_path).convert('RGB')
        return self.transform(image), image_path, T.ToTensor()(image)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--th", type=float, default=0.68)
    parser.add_argument("--metric", type=str, default="pope")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    question_path, image_dir = load_config(args.metric, args.dataset)

    image_list = load_image_list(question_path)

    dataset = ImageDataset(image_dir, image_list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ram_plus(
        pretrained="pretrained/ram_plus_swin_large_14m.pth",
        image_size=384,
        vit='swin_l',
        threshold=args.th
    ).to(device).eval()

    results_dict_ls = []
    for image_tensors, image_paths, _ in dataloader:
        image_tensors = image_tensors.to(device)
        tags = inference(image_tensors, model)[0]
        tags = tags.split(' | ')
        results_dict_ls.append({"image": image_paths[0], "objects": tags})

    with open(args.save_path, "w") as f:
        json.dump(results_dict_ls, f, indent=4)
    print(f"Results saved to {args.save_path}")
