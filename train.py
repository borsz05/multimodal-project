import torch
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm

from src.config.config import paths
from src.data.datasets import Flickr30kDataset, create_dataloader, load_flickr30k_annotations
from src.preprocessing.image_preprocessing import get_image_transform
from src.preprocessing.text_preprocessing import build_vocab, get_text_transform, VocabTokenizer
from src.models import (
    create_contrastive_model,
    contrastive_loss,
    compute_topk_accuracy,
)

#Loads captions, builds vocabulary, and returns the tokenizer object
def build_tokenizer_from_annotations(min_freq: int = 2, max_vocab_size: int | None = 20000): 
    ann_path = paths.flickr30k_annotations_dir / "annotations.csv"
    samples = load_flickr30k_annotations(ann_path)
    captions = [cap for _, cap in samples]
    vocab = build_vocab(captions, min_freq=min_freq, max_size=max_vocab_size)
    tokenizer = VocabTokenizer(vocab)
    return tokenizer, vocab


#Runs model in eval mode, computes average loss and top-k accuracy over the dataset
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    top1_sum = 0.0
    top5_sum = 0.0

    with torch.no_grad():
        for images, token_ids, attn_mask in dataloader:
            images = images.to(device)
            token_ids = token_ids.to(device)
            attn_mask = attn_mask.to(device)

            img_emb, txt_emb, logit_scale = model(images, token_ids, attn_mask)
            loss = contrastive_loss(img_emb, txt_emb, logit_scale)

            logits = logit_scale * img_emb @ txt_emb.t()
            top1 = compute_topk_accuracy(logits, k=1)
            top5 = compute_topk_accuracy(logits, k=5)

            total_loss += loss.item()
            top1_sum += top1
            top5_sum += top5
            total_batches += 1

    return {
        "loss": total_loss / total_batches,
        "top1": top1_sum / total_batches,
        "top5": top5_sum / total_batches,
    }


#Runs one training epoch: forward pass, compute loss, backprop, optimizer step
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, token_ids, attn_mask in tqdm(dataloader, desc="Training"):
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


#Sets up transforms, datasets, model, optimizer, and runs the full training + validation loop
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Tokenizer + text transform --- Builds vocabulary and prepares text preprocessing (tokenization + padding)
    tokenizer, vocab = build_tokenizer_from_annotations(min_freq=2, max_vocab_size=20000)
    text_transform = get_text_transform(tokenizer, max_len=32)

    # --- Image transform ---Creates image preprocessing pipeline for the CNN backbone
    image_transform = get_image_transform()

    # --- Dataset ---Loads Flickr30k images + captions with transforms
    full_dataset = Flickr30kDataset(
        image_transform=image_transform,
        text_transform=text_transform,
        max_samples=None,  # use all available
    )

    # Train/val split (90/10) Splits dataset into 90% train and 10% validation
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = create_dataloader(train_ds, batch_size=32, shuffle=True) #Creates mini-batches for training and validation
    val_loader = create_dataloader(val_ds, batch_size=32, shuffle=False)

    # --- Model ---Builds CLIP-style dual encoder (image + text) with projection heads
    model = create_contrastive_model(
        vocab_size=len(vocab),
        text_embed_dim=256,
        text_proj_dim=256,
        image_proj_dim=256,
        pretrained_backbone=True,
        padding_idx=0,
    )
    model.to(device)

    # --- Optimizer ---AdamW optimizer for training all model parameters
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    # --- Train loop ---Runs epochs; after each: train → validate → save best model
    num_epochs = 3
    best_val_loss = float("inf")
    save_path = paths.models_dir
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_loss: {val_metrics['loss']:.4f} "
            f"- val_top1: {val_metrics['top1']:.3f} "
            f"- val_top5: {val_metrics['top5']:.3f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            model_file = save_path / "flickr30k_clip_best.pth"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": vocab,
                },
                model_file,
            )
            print(f"Saved best model to: {model_file}")


if __name__ == "__main__":
    main()
