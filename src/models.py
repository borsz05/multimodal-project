import torch
import torch.nn as nn
import timm


class ImageEncoder(nn.Module): # Extracts a feature vector from the image using a CNN backbone
    """
    EfficientNet (vagy más timm backbone) feature extractor.
    Nem ad közvetlen class logitot, csak egy feature vektort.
    """

    def __init__(self, backbone_name: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.out_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W]
        return: [B, out_dim]
        """
        return self.backbone(x)


class TextEncoder(nn.Module): #Embeds tokens, applies masked mean pooling, projects to 256 dims
    """
    Token ID -> embedding -> masked mean pool -> projection.
    padding_idx biztosítja, hogy a PAD token ne befolyásolja az átlagot.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        proj_dim: int = 256,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=padding_idx,
        )
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.padding_idx = padding_idx
        self.out_dim = proj_dim

    def forward(self, token_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [B, L] int64
        attn_mask: [B, L] float, 1 for real tokens, 0 for pad
        return: [B, out_dim]
        """
        embeds = self.embedding(token_ids)  # [B, L, E]
        mask = attn_mask.unsqueeze(-1)  # [B, L, 1]
        summed = (embeds * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / denom
        return self.projection(pooled)


class ProjectionHead(nn.Module): #Maps image/text features into the same normalized embedding space
    """Linear projection + L2 norm a közös embedding térhez."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return nn.functional.normalize(x, dim=-1)


class CLIPLikeModel(nn.Module):
    """
    Multimodális kontrasztív modell (CLIP-szerű):
    - Kép encoder (timm backbone)
    - Szöveg encoder (embedding + masked mean pool)
    - Közös embedding és skálázott dot-product loss
    """

    def __init__(
        self,
        image_backbone: str = "efficientnet_b0",
        vocab_size: int = 10000,
        text_embed_dim: int = 256,
        text_proj_dim: int = 256,
        image_proj_dim: int = 256,
        pretrained_backbone: bool = True,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(
            backbone_name=image_backbone,
            pretrained=pretrained_backbone,
        )
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            proj_dim=text_proj_dim,
            padding_idx=padding_idx,
        )
        self.image_head = ProjectionHead(self.image_encoder.out_dim, image_proj_dim)
        self.text_head = ProjectionHead(self.text_encoder.out_dim, image_proj_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * 1.0)

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor, attn_mask: torch.Tensor):
        """
        Returns L2-normalized embeddings and logit_scale.
        """
        img_feat = self.image_encoder(images)
        txt_feat = self.text_encoder(token_ids, attn_mask)
        img_emb = self.image_head(img_feat)
        txt_emb = self.text_head(txt_feat)
        logit_scale = self.logit_scale.exp()
        return img_emb, txt_emb, logit_scale


def contrastive_loss(image_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale: torch.Tensor):
    """
    InfoNCE-szerű veszteség: egy batch-en belül a helyes (i,i) párokat kell megtalálni.
    """
    logits_per_image = logit_scale * image_emb @ text_emb.t()
    logits_per_text = logits_per_image.t()
    targets = torch.arange(image_emb.size(0), device=image_emb.device)
    loss_i = nn.functional.cross_entropy(logits_per_image, targets)
    loss_t = nn.functional.cross_entropy(logits_per_text, targets)
    return (loss_i + loss_t) / 2


def compute_topk_accuracy(logits: torch.Tensor, k: int = 5):
    targets = torch.arange(logits.size(0), device=logits.device)
    _, preds = logits.topk(k, dim=1)
    correct = (preds == targets.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()


def create_contrastive_model(
    image_backbone: str = "efficientnet_b0",
    vocab_size: int = 10000,
    text_embed_dim: int = 256,
    text_proj_dim: int = 256,
    image_proj_dim: int = 256,
    pretrained_backbone: bool = True,
    padding_idx: int = 0,
) -> CLIPLikeModel:
    return CLIPLikeModel(
        image_backbone=image_backbone,
        vocab_size=vocab_size,
        text_embed_dim=text_embed_dim,
        text_proj_dim=text_proj_dim,
        image_proj_dim=image_proj_dim,
        pretrained_backbone=pretrained_backbone,
        padding_idx=padding_idx,
    )
