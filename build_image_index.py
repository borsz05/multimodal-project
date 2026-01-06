from pathlib import Path
import torch
from tqdm import tqdm
from PIL import Image

from transformers import BertTokenizerFast

from src.config.config import paths
from src.dataset.datasets import load_flickr30k_annotations
from src.preprocessing.image_preprocessing import get_image_transform
from src.preprocessing.text_preprocessing import get_bert_text_transform
from src.models import create_contrastive_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Load model ---
    model_file = paths.models_dir / "flickr30k_bert_clip_best.pth"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    checkpoint = torch.load(model_file, map_location=device)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

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

    # --- Data: only need to collect the image names ---
    ann_path = paths.flickr30k_annotations_dir / "annotations.csv"
    samples = load_flickr30k_annotations(ann_path)
    image_names = sorted({img_name for (img_name, _) in samples})

    images_dir = paths.flickr30k_images_dir
    transform = get_image_transform()

    all_embs = []
    all_paths = []

    batch_size = 64
    print(f"Building image index for {len(image_names)} unique images...")

    with torch.no_grad():
        for i in tqdm(range(0, len(image_names), batch_size), desc="Indexing images"):
            batch_names = image_names[i : i + batch_size]
            imgs = []
            valid_names = []

            for name in batch_names:
                img_path = images_dir / name
                if not img_path.exists():
                    continue
                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        imgs.append(transform(img))
                        valid_names.append(name)
                except Exception as e:
                    print(f"Failed to load image {img_path}: {e}")

            if not imgs:
                continue

            batch_tensor = torch.stack(imgs, dim=0).to(device)
            img_feat = model.image_encoder(batch_tensor)
            img_emb = model.image_head(img_feat)  # [B, D]
            all_embs.append(img_emb.cpu())
            all_paths.extend(valid_names)

    if not all_embs:
        raise RuntimeError("No image embeddings computed. Check your images directory.")

    all_embs_tensor = torch.cat(all_embs, dim=0)  # [N, D]
    index_path = paths.models_dir / "flickr30k_image_index.pt"
    torch.save(
        {
            "embeddings": all_embs_tensor,
            "image_names": all_paths,
        },
        index_path,
    )
    print(f"Saved image index to: {index_path}")
    print(f"Total indexed images: {len(all_paths)}")


if __name__ == "__main__":
    main()
