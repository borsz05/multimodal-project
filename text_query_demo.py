import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from transformers import BertTokenizerFast

from src.config.config import paths
from src.preprocessing.image_preprocessing import get_image_transform
from src.preprocessing.text_preprocessing import get_bert_text_transform
from src.models import create_contrastive_model


def load_model_and_index(device):
    # Model
    model_file = paths.models_dir / "flickr30k_bert_clip_best.pth"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    checkpoint = torch.load(model_file, map_location=device)

    model = create_contrastive_model(
        image_backbone="efficientnet_b0",
        vocab_size=None,
        image_proj_dim=256,
        pretrained_backbone=True,
        padding_idx=0,
        use_bert=True,
        bert_model_name=checkpoint.get("bert_model_name", "bert-base-uncased"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Tokenizer + text_transform
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    text_transform = get_bert_text_transform(tokenizer, max_len=32)

    # Image index
    index_path = paths.models_dir / "flickr30k_image_index.pt"
    if not index_path.exists():
        raise FileNotFoundError(f"Image index not found: {index_path}")
    index_data = torch.load(index_path, map_location="cpu")
    image_embs = index_data["embeddings"]  # [N, D]
    image_names = index_data["image_names"]

    return model, text_transform, image_embs, image_names


def show_topk_images(image_names, scores, k=5):
    images_dir = paths.flickr30k_images_dir
    k = min(k, len(image_names))

    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
    if k == 1:
        axes = [axes]

    for ax, name, score in zip(axes, image_names[:k], scores[:k]):
        img_path = images_dir / name
        with Image.open(img_path) as img:
            ax.imshow(img.convert("RGB"))
        ax.axis("off")
        ax.set_title(f"{name}\n{score * 100:.1f}%", fontsize=9)

    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model, text_transform, image_embs, image_names = load_model_and_index(device)
    image_embs = image_embs.to(device, dtype=torch.float32)

    print("Text to image search. Submit an empty line to exit.\n")

    while True:
        query = input("Enter a text description (press ENTER to exit): ").strip()
        if not query:
            break

        # Text embedding
        token_ids, attn_mask = text_transform(query)
        token_ids = token_ids.unsqueeze(0).to(device)     # [1, L]
        attn_mask = attn_mask.unsqueeze(0).to(device)     # [1, L]

        with torch.no_grad():
            # The image embeddings are already indexed; we only compute the text side here
            txt_feat = model.text_encoder(token_ids, attn_mask)
            txt_emb = model.text_head(txt_feat)  # [1, D]
            txt_emb = txt_emb.squeeze(0).float()  # [D]

        # Cosine similarity = dot product because they are L2-normalized
        sims = torch.matmul(image_embs, txt_emb)  # [N]
        logit_scale = model.logit_scale.exp()
        temperature = 5.0
        scaled_sims = sims * logit_scale * temperature

        # Top-5 indices (based on raw similarity)
        k = 5
        topk_vals, topk_idx = torch.topk(scaled_sims, k=k)

        # Softmax-normalized values only over the top-5 images
        topk_probs = torch.softmax(topk_vals, dim=0)

        topk_vals = topk_probs.cpu().tolist()
        topk_names = [image_names[i] for i in topk_idx.cpu().tolist()]

        print("\nTop-5 results:")
        for rank, (name, score) in enumerate(zip(topk_names, topk_vals), start=1):
            print(f"{rank}. {name} ({score * 100:.1f}%)")

        # Display images
        show_topk_images(topk_names, topk_vals, k=k)
        print("\n---\n")


if __name__ == "__main__":
    main()
