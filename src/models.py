import torch
import torch.nn as nn
import timm


class ImageEncoder(nn.Module):
    """
    EfficientNet (vagy más timm backbone) feature extractor.
    Nem ad közvetlen class logitot, csak egy feature vektort.
    """

    def __init__(self, backbone_name: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        # num_classes=0 => timm csak feature-t ad vissza
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        # kimeneti dimenzió (pl. 1280 efficientnet_b0-nál)
        self.out_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W]
        return: [B, out_dim]
        """
        return self.backbone(x)


class TextEncoder(nn.Module):
    """
    Egyszerű MLP a szöveg embeddingre.
    Most feltételezzük, hogy egy fix-dimenziós embeddinget kapunk (pl. 300 dim).
    Később ide jöhet LSTM / BERT is.
    """

    def __init__(self, text_dim: int = 300, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, text_dim]
        return: [B, hidden_dim]
        """
        return self.net(x)


class MultimodalClassifier(nn.Module):
    """
    Kép + szöveg → közös embedding → osztályozás.
    """

    def __init__(
        self,
        num_classes: int,
        image_backbone: str = "efficientnet_b0",
        text_dim: int = 300,
        text_hidden_dim: int = 256,
        fusion_hidden_dim: int = 512,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(
            backbone_name=image_backbone,
            pretrained=pretrained_backbone,
        )
        self.text_encoder = TextEncoder(
            text_dim=text_dim,
            hidden_dim=text_hidden_dim,
        )

        fusion_input_dim = self.image_encoder.out_dim + self.text_encoder.out_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, H, W]
        text_embeddings: [B, text_dim]
        return: [B, num_classes] (logits)
        """
        img_feat = self.image_encoder(images)
        txt_feat = self.text_encoder(text_embeddings)
        fused = torch.cat([img_feat, txt_feat], dim=1)
        logits = self.classifier(fused)
        return logits


def create_multimodal_model(
    num_classes: int,
    image_backbone: str = "efficientnet_b0",
    text_dim: int = 300,
    text_hidden_dim: int = 256,
    fusion_hidden_dim: int = 512,
    pretrained_backbone: bool = True,
) -> MultimodalClassifier:
    """
    Helper függvény a modell példányosítására.
    """
    model = MultimodalClassifier(
        num_classes=num_classes,
        image_backbone=image_backbone,
        text_dim=text_dim,
        text_hidden_dim=text_hidden_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        pretrained_backbone=pretrained_backbone,
    )
    return model
