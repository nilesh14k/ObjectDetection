import os
import torch
from PIL import Image


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "labels"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.root, "labels", self.labels[idx])

        # Load the image and its labels
        img = Image.open(img_path).convert("RGB")
        with open(label_path, "r") as f:
            labels = []
            for line in f.readlines():
                label = line.strip().split()
                bbox = [int(float(label[i])) for i in range(1, 5)]
                label_id = int(label[0])
                labels.append({"bbox": bbox, "label": label_id})

        # Apply transforms
        if self.transforms is not None:
            img, labels = self.transforms(img, labels)

        # Convert labels to PyTorch tensors
        boxes = [torch.tensor(label["bbox"], dtype=torch.float32) for label in labels]
        labels = [torch.tensor(label["label"], dtype=torch.int64) for label in labels]
        target = {"boxes": torch.stack(boxes), "labels": torch.stack(labels)}

        return img, target

    def __len__(self):
        return len(self.imgs)