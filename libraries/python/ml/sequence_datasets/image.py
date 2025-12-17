import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ConvertImageDtype
from torchvision.io import read_image


class ImageDataset(Dataset):
    def __init__(self, folder: Path, target_size=(1080, 1920)):
        self.paths = sorted(
            [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ],
        )
        self.resize = Resize(target_size)
        self.convert = ConvertImageDtype(torch.float32)  # Converts to [0,1]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = read_image(self.paths[idx])
        img = self.convert(img)
        img = self.resize(img)
        return img
