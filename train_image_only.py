"""
Egyszerű kontrasztív tanítás ugyanazzal az image-caption párral,
de a futás végén csak a kép-encoder súlyát mentjük ki.
"""

import torch
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm

from src.config.config import paths
from src.data.datasets import Flickr30kDataset, create_dataloader, load_flickr30k_annotations
from src.preprocessing.image_preprocessing import get_image_transform
from src.preprocessing.text_preprocessing import build_vocab, get_text_transform, VocabTokenizer
from src.models import create_contrastive_model, contrastive_loss

# "Builds tokenizer and vocabulary from annotation captions."
def build_tokenizer_from_annotations(min_freq: int = 2, max_vocab_size: int | None = 20000):
    ann_path = paths.flickr30k_annotations_dir / "annotations.csv"
    samples = load_flickr30k_annotations(ann_path)
    captions = [cap for _, cap in samples]
    vocab = build_vocab(captions, min_freq=min_freq, max_size=max_vocab_size)
    tokenizer = VocabTokenizer(vocab)
    return tokenizer, vocab

# "Runs one epoch of contrastive training."
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, token_ids, attn_mask in tqdm(dataloader, desc="Training (contrastive)"):
        images = images.to(device)
        token_ids = token_ids.to(device)
        attn_mask = attn_mask.to(device)

        optimizer.zero_grad()
        img_emb, txt_emb, logit_scale = model(images, token_ids, attn_mask)
        loss = contrastive_loss(img_emb, txt_emb, logit_scale)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# "Main training routine: trains model and saves only image encoder."
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer, vocab = build_tokenizer_from_annotations(min_freq=2, max_vocab_size=20000)
    text_transform = get_text_transform(tokenizer, max_len=32)
    image_transform = get_image_transform()

    full_dataset = Flickr30kDataset(
        image_transform=image_transform,
        text_transform=text_transform,
        max_samples=None,
    )

    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, _ = random_split(full_dataset, [train_size, val_size])
    train_loader = create_dataloader(train_ds, batch_size=32, shuffle=True)

    model = create_contrastive_model(
        vocab_size=len(vocab),
        text_embed_dim=256,
        text_proj_dim=256,
        image_proj_dim=256,
        pretrained_backbone=True,
        padding_idx=0,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    num_epochs = 2
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, loss: {loss:.4f}")

    save_dir = paths.models_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # "Saves only the image encoder weights."
    torch.save(
        {
            "image_encoder_state": model.image_encoder.state_dict(),
            "image_head_state": model.image_head.state_dict(),
        },
        save_dir / "image_encoder_from_clip.pth",
    )
    print(f"Saved image encoder weights to {save_dir / 'image_encoder_from_clip.pth'}")


if __name__ == "__main__":
    main()
