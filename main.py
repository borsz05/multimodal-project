from src.config.config import paths
from src.dataset.datasets import Flickr30kDataset, create_dataloader, load_flickr30k_annotations
from src.preprocessing.image_preprocessing import get_image_transform
from src.preprocessing.text_preprocessing import build_vocab, get_text_transform, VocabTokenizer


def sanity_check():
    print("Project root:", paths.project_root)
    print("Flickr30k dir:", paths.flickr30k_dir)

    ann_path = paths.flickr30k_annotations_dir / "annotations.csv"
    samples = load_flickr30k_annotations(ann_path)
    print(f"Összes annotációs sor: {len(samples)}")

    captions = [cap for _, cap in samples]
    vocab = build_vocab(captions, min_freq=2, max_size=5000)
    tokenizer = VocabTokenizer(vocab)
    text_transform = get_text_transform(tokenizer, max_len=30)

    dataset = Flickr30kDataset(
        image_transform=get_image_transform(),
        text_transform=text_transform,
        max_samples=32,
    )
    print("Flickr30kDataset hossza:", len(dataset))

    image, token_ids, attn_mask = dataset[0]
    print("Egy minta image shape:", image.shape)
    print("Egy minta token_ids shape:", token_ids.shape)
    print("Egy minta attn_mask shape:", attn_mask.shape)

    dataloader = create_dataloader(dataset, batch_size=4)
    images, token_batch, attn_batch = next(iter(dataloader))
    print("Batch images shape:", images.shape)
    print("Batch token_ids shape:", token_batch.shape)
    print("Batch attention_mask shape:", attn_batch.shape)


if __name__ == "__main__":
    sanity_check()
