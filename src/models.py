import torch
import torch.nn as nn
import timm
from transformers import BertModel


class ImageEncoder(nn.Module):  # Extracts a feature vector from the image using a CNN backbone
    """
    EfficientNet (or another timm backbone) feature extractor.
    Does not output class logits directly, only a feature vector.
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


class TextEncoder(nn.Module):  # Embeds tokens, applies masked mean pooling, projects to 256 dims
    """
    Token ID -> embedding -> masked mean pool -> projection.
    padding_idx ensures that the PAD token does not influence the average.
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


class BertTextEncoder(nn.Module):
    """
    BERT-based text encoder.
    We use the hidden state of the CLS token as the text embedding.
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.out_dim = self.bert.config.hidden_size  # e.g. 768

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, L]
        attention_mask: [B, L]
        return: [B, out_dim]
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # CLS token
        return cls


class ProjectionHead(nn.Module):  # Maps image/text features into the same normalized embedding space
    """Linear projection + L2 normalization to a shared embedding space."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return nn.functional.normalize(x, dim=-1)


class CLIPLikeModel(nn.Module):
    """
    Multimodal contrastive model (CLIP-like):
    - Image encoder (timm backbone)
    - Text encoder (embedding + masked mean pool)
    - Shared embedding space and scaled dot-product loss
    """

    def __init__(
        self,
        image_backbone: str = "efficientnet_b0",
        vocab_size: int | None = None,
        text_embed_dim: int = 256,
        text_proj_dim: int = 256,
        image_proj_dim: int = 256,
        pretrained_backbone: bool = True,
        padding_idx: int = 0,
        use_bert: bool = False,
        bert_model_name: str = "bert-base-uncased",
    ):
        super().__init__()

        # --- Image encoder ---
        self.image_encoder = ImageEncoder(
            backbone_name=image_backbone,
            pretrained=pretrained_backbone,
        )

        # --- Text encoder: either custom or BERT ---
        if use_bert:
            self.text_encoder = BertTextEncoder(model_name=bert_model_name)
        else:
            if vocab_size is None:
                raise ValueError("vocab_size must be provided when not using BERT text encoder.")
            self.text_encoder = TextEncoder(
                vocab_size=vocab_size,
                embed_dim=text_embed_dim,
                proj_dim=text_proj_dim,
                padding_idx=padding_idx,
            )

        # --- Projection heads ---
        self.image_head = ProjectionHead(self.image_encoder.out_dim, image_proj_dim)
        self.text_head = ProjectionHead(self.text_encoder.out_dim, image_proj_dim)

        # --- Logit scale ---
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
    InfoNCE-like loss: within a batch, the correct (i, i) pairs must be identified.
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
    vocab_size: int | None = None,
    text_embed_dim: int = 256,
    text_proj_dim: int = 256,
    image_proj_dim: int = 256,
    pretrained_backbone: bool = True,
    padding_idx: int = 0,
    use_bert: bool = False,
    bert_model_name: str = "bert-base-uncased",
) -> CLIPLikeModel:
    return CLIPLikeModel(
        image_backbone=image_backbone,
        vocab_size=vocab_size,
        text_embed_dim=text_embed_dim,
        text_proj_dim=text_proj_dim,
        image_proj_dim=image_proj_dim,
        pretrained_backbone=pretrained_backbone,
        padding_idx=padding_idx,
        use_bert=use_bert,
        bert_model_name=bert_model_name,
    )
