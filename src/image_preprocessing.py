from torchvision import transforms


def get_image_transform(image_size: int = 224):
    """
    Preprocess a Flickr30k image for EfficientNet / CNN:
    - resize + center crop
    - tensor konverzió
    - ImageNet normalizálás
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet
            std=[0.229, 0.224, 0.225],
        ),
    ])
