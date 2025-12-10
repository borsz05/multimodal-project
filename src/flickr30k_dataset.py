import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np

from src.config import paths
from src.datasets import load_flickr30k_annotations

class Flickr30kDataset(Dataset):
    """
    Flickr30k Dataset
    -----------------
    - Betölti a CSV annotációkat (image_name + caption)
    - Képeket a mappából
    - Paraméterként hívja a kép- és szöveg preprocessing függvényeket
    - Minden caption külön minta → 1 kép akár többször szerepelhet
    """

    def __init__(self, image_transform, text_transform):
        """
        :param image_transform: függvény, ami egy PIL Image-ből tensor-t készít
        :param text_transform: függvény, ami egy caption stringből tensor-t készít
        """
        self.images_dir = paths.flickr30k_images_dir
        self.ann_path = paths.flickr30k_annotations_dir / "annotations.csv"

        # Betöltjük a CSV fájlt
        self.samples = load_flickr30k_annotations(self.ann_path)

        # Preprocessing függvények
        self.image_transform = image_transform
        self.text_transform = text_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Visszaadja az idx-edik mintát:
        - feldolgozott kép tensor
        - feldolgozott caption tensor
        - placeholder címke (pl. 0)
        """
        image_name, caption = self.samples[idx]
        image_path = self.images_dir / image_name

        # --- Kép betöltés ---
        image = Image.open(image_path).convert("RGB")

        # --- Kép preprocessing ---
        # Pl. resize, normalize, to tensor
        image = self.image_transform(image)

        # --- Szöveg preprocessing ---
        # Pl. tokenizálás, indexálás, padding
        text_tensor = self.text_transform(caption)

        # --- Placeholder label ---
        label = 0  # később a valós címkére cserélhető

        return image, text_tensor, label


# ========================================================
# Dataloader helper
# ========================================================
def create_dataloader(dataset, batch_size=32, shuffle=True):
    """
    Egyszerű wrapper a PyTorch DataLoader fölé
    """
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)