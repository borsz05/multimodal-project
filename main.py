import torch

from src.config import paths
from src.datasets import DummyDataset, create_dataloader
from src.models import create_multimodal_model


def main():
    print("Project root:", paths.project_root)
    print("Flickr30k dir:", paths.flickr30k_dir)

    # Eszköz kiválasztása (ha van CUDA, azt használja)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dummy dataset – 10 minta, 10 osztály
    dataset = DummyDataset(size=10, num_classes=10)
    dataloader = create_dataloader(dataset, batch_size=2)

    # Modell létrehozása
    model = create_multimodal_model(num_classes=10)
    model.to(device)
    model.eval()  # most csak forwardot tesztelünk

    # Egy batch teszt forward
    print("\nForward pass teszt:")
    with torch.no_grad():
        for images, texts, labels in dataloader:
            images = images.to(device)
            texts = texts.to(device)
            logits = model(images, texts)
            print("images shape:", images.shape)
            print("texts shape:", texts.shape)
            print("logits shape:", logits.shape)
            break


if __name__ == "__main__":
    main()
