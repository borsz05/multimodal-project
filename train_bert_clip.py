import torch
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
from transformers import BertTokenizerFast

from src.config.config import paths
from src.dataset.datasets import Flickr30kDataset, create_dataloader
from src.preprocessing.image_preprocessing import get_image_transform
from src.preprocessing.text_preprocessing import get_bert_text_transform
from src.models import create_contrastive_model, contrastive_loss, compute_topk_accuracy


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


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, token_ids, attn_mask in tqdm(dataloader, desc="Training (BERT-CLIP)"):
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- BERT tokenizer + text transform ---
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    text_transform = get_bert_text_transform(tokenizer, max_len=32)

    # --- Image transform ---
    image_transform = get_image_transform()

    # --- Dataset ---
    full_dataset = Flickr30kDataset(
        image_transform=image_transform,
        text_transform=text_transform,
        max_samples=100000,
    )

    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = create_dataloader(train_ds, batch_size=64, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=64, shuffle=False)

    # --- Modell: BERT-es CLIP ---
    model = create_contrastive_model(
        image_backbone="efficientnet_b0",
        vocab_size=None,          # BERT-nél nem használjuk
        image_proj_dim=256,
        pretrained_backbone=True,
        padding_idx=0,
        use_bert=True,
        bert_model_name="bert-base-uncased",
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    num_epochs = 5
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
            model_file = save_path / "flickr30k_bert_clip_best.pth"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "bert_model_name": "bert-base-uncased",
                },
                model_file,
            )
            print(f"Saved best BERT-CLIP model to: {model_file}")


if __name__ == "__main__":
    main()
