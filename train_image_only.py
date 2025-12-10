import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm

from src.config import paths
from src.datasets import Flickr30kDataset, create_dataloader
from src.models import create_image_only_model


class Flickr30kImageOnlyDebug(Dataset):
    """
    Wrapper: Flickr30kDataset -> (image, label)
    Itt a label egy DUMMY, hash alapú osztály:
    label = hash(image_name) % num_classes

    Ez csak technikai teszt, hogy a CNN tréning pipeline működik.
    Valódi projektben ezt kicserélitek egy értelmes címkés feladatra.
    """

    def __init__(self, base_dataset: Flickr30kDataset, num_classes: int = 10, max_samples: int | None = 5000):
        self.base_dataset = base_dataset
        self.num_classes = num_classes

        # opcionálisan limitáljuk a minták számát, hogy ne legyen nagyon lassú CPU-n
        if max_samples is not None:
            self.indices = list(range(min(max_samples, len(base_dataset))))
        else:
            self.indices = list(range(len(base_dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        image, caption = self.base_dataset[base_idx]

        # image_name-ből stabil dummy labelt generálunk
        image_name, _ = self.base_dataset.samples[base_idx]
        label = hash(image_name) % self.num_classes

        return image, label


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(dataloader, desc="Training (image-only)"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    print("Project root:", paths.project_root)
    print("Flickr30k dir:", paths.flickr30k_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # === Alap Flickr30k dataset (image + caption string) ===
    base_dataset = Flickr30kDataset()

    # === Csak képes debug dataset (image + dummy label) ===
    num_classes = 10
    debug_dataset = Flickr30kImageOnlyDebug(base_dataset, num_classes=num_classes, max_samples=5000)

    dataloader = create_dataloader(debug_dataset, batch_size=32)

    # === Modell ===
    model = create_image_only_model(num_classes=num_classes)
    model.to(device)

    # === Loss + optimizer ===
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # === Train loop ===
    num_epochs = 2
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"[Image-only] Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    # === Modell mentése ===
    save_dir = paths.models_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "image_only_cnn_debug.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[Image-only] Model saved to: {save_path}")


if __name__ == "__main__":
    main()
