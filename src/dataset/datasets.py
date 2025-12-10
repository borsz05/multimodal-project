import csv
from pathlib import Path
from typing import Callable, List, Tuple

from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.config.config import paths
from src.preprocessing.image_preprocessing import get_image_transform


def load_flickr30k_annotations(csv_path: Path) -> List[Tuple[str, str]]:
    """
    Load Flickr30k-style annotations.
    Expected format (no header):
        image_name|comment_number|comment
    Returns a list of (image_name, caption) tuples.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        sep="|",
        header=None,
        names=["image_name", "comment_number", "comment"],
        dtype=str,
        encoding="utf-8",
        engine="python",
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
    )

    df["image_name"] = df["image_name"].astype(str).str.strip().str.strip('"')
    df["comment"] = df["comment"].astype(str).str.strip(' ,"')

    df = df.dropna(subset=["image_name", "comment"])
    df = df[(df["image_name"] != "") & (df["comment"] != "")]

    samples: List[Tuple[str, str]] = list(
        df[["image_name", "comment"]].itertuples(index=False, name=None)
    )
    return samples


class Flickr30kDataset(Dataset):
    """
    Flickr30k dataset that returns preprocessed tensors:
    image_tensor, token_ids, attention_mask
    """

    def __init__(
        self,
        image_transform: Callable[[Image.Image], torch.Tensor] | None = None,
        text_transform: Callable[[str], Tuple[torch.Tensor, torch.Tensor]] | None = None,
        max_samples: int | None = None,
    ):
        self.images_dir = paths.flickr30k_images_dir
        self.ann_path = paths.flickr30k_annotations_dir / "annotations.csv"

        raw_samples = load_flickr30k_annotations(self.ann_path)
        samples: List[Tuple[str, str]] = []
        for image_name, caption in raw_samples:
            if (self.images_dir / image_name).exists():
                samples.append((image_name, caption))

        if max_samples is not None:
            samples = samples[:max_samples]

        self.samples = samples
        self.image_transform = image_transform or get_image_transform()
        self.text_transform = text_transform or (lambda caption: caption)

        print(
            f"Flickr30kDataset: {len(self.samples)} usable samples "
            f"(annotations: {len(raw_samples)}, images missing: {len(raw_samples) - len(self.samples)})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_name, caption = self.samples[idx]
        image_path = self.images_dir / image_name

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image_tensor = self.image_transform(img)

        token_ids, attn_mask = self.text_transform(caption)
        return image_tensor, token_ids, attn_mask


def create_dataloader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
