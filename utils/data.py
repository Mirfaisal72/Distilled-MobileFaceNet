import os
from typing import Tuple, List
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AlignedFaceDataset(Dataset):
    """
    Loads aligned 112x112 face images from a root folder with structure:
    root/
      person_a/*.png|jpg
      person_b/*.png|jpg
    Returns (image_tensor, label, path).
    """

    def __init__(self, root: str, transform=None, extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        self.root = root
        self.transform = transform or default_transform()
        self.extensions = extensions
        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx = {}
        self.classes = []

        self._index()

    def _index(self):
        people = [d for d in sorted(os.listdir(self.root)) if os.path.isdir(os.path.join(self.root, d))]
        self.classes = people
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for person in people:
            pdir = os.path.join(self.root, person)
            for fname in sorted(os.listdir(pdir)):
                if any(fname.lower().endswith(ext) for ext in self.extensions):
                    self.samples.append((os.path.join(pdir, fname), self.class_to_idx[person]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path


def default_transform():
    # Augmentations to reduce overfitting and improve robustness
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # scale to [-1,1]
    ])
