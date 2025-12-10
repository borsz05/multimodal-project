import torch

from src.config import paths
from src.datasets import DummyDataset, create_dataloader, load_flickr30k_annotations, Flickr30kDataset
from src.models import create_multimodal_model


def main():
    print("Project root:", paths.project_root)
    print("Flickr30k dir:", paths.flickr30k_dir)

    # ---- 1) Annotáció teszt (ez már megy) ----
    ann_path = paths.flickr30k_annotations_dir / "annotations.csv"
    print("\nAnnotációs fájl elérési útja:", ann_path)
    samples = load_flickr30k_annotations(ann_path)
    print(f"Összes minta (sor) az annotációkban: {len(samples)}")

    # ---- 2) Flickr30kDataset teszt ----
    dataset = Flickr30kDataset()
    print("Flickr30kDataset hossza:", len(dataset))

    image, caption = dataset[0]
    print("Egy minta image shape:", image.shape)
    print("Egy minta caption:", caption)

    # Dataloader teszt
    dataloader = create_dataloader(dataset, batch_size=4)
    images, captions = next(iter(dataloader))
    print("Batch images shape:", images.shape)
    print("Batch captions (lista hossza):", len(captions))


if __name__ == "__main__":
    main()
