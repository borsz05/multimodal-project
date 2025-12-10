import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from PIL import Image
import numpy as np
import pandas as pd

from src.config import paths


def load_flickr30k_annotations(csv_path: Path):
    """
    Flickr30k jellegű annotációs fájl betöltése.
    Formátum (fejléc NÉLKÜL):
        image_name| comment_number| comment

    Példa sor:
        1000092795.jpg| 0| Two young guys ...
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotációs fájl nem található: {csv_path}")

    # NINCS fejléc, minden sor adat -> header=None
    # MINDENT stringként olvasunk, hogy ne legyen dtype-hiba
    df = pd.read_csv(
        csv_path,
        sep="|",
        header=None,
        names=["image_name", "comment_number", "comment"],
        dtype=str,
        encoding="utf-8",
        engine="python",
        quotechar="¶",        # olyan karakter, ami biztosan NINCS a fájlban
        on_bad_lines="skip"
        )

    # whitespace-ek eltávolítása
    df["image_name"] = df["image_name"].astype(str).str.strip()
    df["comment"] = df["comment"].astype(str).str.strip()

    # üres caption vagy image_name sorok kiszűrése
    df = df.dropna(subset=["image_name", "comment"])
    df = df[(df["image_name"] != "") & (df["comment"] != "")]

    # csak az image_name + comment párok kellenek
    samples = list(df[["image_name", "comment"]].itertuples(index=False, name=None))

    return samples


# ============================================================
# 1) DUMMY DATASET – hogy a training pipeline működjön
# ============================================================

class DummyDataset(Dataset):
    """
    Egy teljesen egyszerű dataset:
    - 224x224 random képeket generál
    - random (dummy) text embeddinget ad mellé
    - random 0..9 közötti címkével
    Ez csak arra jó, hogy leteszteljük a modellt és a train loopot.
    """

    def __init__(self, size=1000, num_classes=10):
        self.size = size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Random kép (3x224x224)
        image = torch.rand(3, 224, 224)

        # Random szöveg-embedding (pl. 300 dimenzió)
        text_embedding = torch.rand(300)

        # Random címke
        label = random.randint(0, self.num_classes - 1)

        return image, text_embedding, label


# ============================================================
# 2) FLICKR30K DATASET (PLACERHOLDER)
# ============================================================

class Flickr30kDataset(Dataset):
    """
    EGYELŐRE ÜRES, csak a struktúra.
    Később:
    - beolvassuk a caption fájlokat (annotations)
    - beolvassuk a képeket (images)
    - tokenizálás
    - EfficientNet pre-processing
    """

    def __init__(self):
        self.images_dir = paths.flickr30k_images_dir
        self.ann_dir = paths.flickr30k_annotations_dir

        # később feltöltjük egy listával: [(image_path, text), ...]
        self.samples = []

        # PLACEHOLDER: 10 darab dummy útvonal
        for i in range(10):
            self.samples.append(("sample_image.jpg", "a sample caption"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, caption = self.samples[idx]

        # Amíg nincs valódi kép → random image
        image = torch.rand(3, 224, 224)

        # Amíg nincs tokenizer → random embedding
        text_embedding = torch.rand(300)

        # PLACEHOLDER címke (pl. 0)
        label = 0

        return image, text_embedding, label


# ============================================================
# 3) DATALOADER HELPER FÜGGVÉNY
# ============================================================

def create_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
