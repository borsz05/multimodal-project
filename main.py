import torch

from src.config import paths
from src.datasets import DummyDataset, create_dataloader, load_flickr30k_annotations
from src.models import create_multimodal_model


def main():
    print("Project root:", paths.project_root)
    print("Flickr30k dir:", paths.flickr30k_dir)

    # ---- 1) Annotációs fájl beolvasás teszt ----
    ann_path = paths.flickr30k_annotations_dir / "annotations.csv"  # ha más a neve, írd át

    print("\nAnnotációs fájl elérési útja:", ann_path)

    samples = load_flickr30k_annotations(ann_path)
    print(f"Összes minta (sor) az annotációkban: {len(samples)}")

    print("\nElső 5 minta:")
    for i, (image_id, caption) in enumerate(samples[:5]):
        print(f"{i+1}. image_id = {image_id} | caption = {caption}")

    # (A korábbi dummy + model forward teszt most akár maradhat is,
    # de ha zavar, ideiglenesen kikommentezheted)

if __name__ == "__main__":
    main()
