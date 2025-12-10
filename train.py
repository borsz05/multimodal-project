import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from src.config import paths
from src.datasets import DummyDataset, create_dataloader
from src.models import create_multimodal_model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, texts, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        texts = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images, texts)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    # ============= GPU / CPU választás =============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ============= Dummy dataset =============
    dataset = DummyDataset(size=200, num_classes=10)
    dataloader = create_dataloader(dataset, batch_size=16)

    # ============= Modell =============
    model = create_multimodal_model(num_classes=10)
    model.to(device)

    # ============= Loss + Optimizer =============
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ============= Train loop =============
    num_epochs = 2
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    # ============= Modell mentése =============
    save_path = paths.models_dir
    save_path.mkdir(parents=True, exist_ok=True)

    model_file = save_path / "dummy_multimodal_model.pth"
    torch.save(model.state_dict(), model_file)

    print(f"Model saved to: {model_file}")


if __name__ == "__main__":
    main()
