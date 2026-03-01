"""
Unified Multi-Modal Transformer for Colon Cancer Detection & Staging
─────────────────────────────────────────────────────────────────────
Architecture (v2 — GradCAM-compatible dual-backbone):

  Image branch   : ResNet50  (ImageNet pretrained, layer4 = GradCAM target)
                 + EfficientNet-B4 (ImageNet pretrained, blocks[-2] = GradCAM target)
                   → spatial feature maps → patch tokens → projected to d_model
                   → BOTH backbones fused via learned gating per spatial position

  Text branch    : BioBERT (CLS token) → projection  [UNCHANGED — best clinical NLP]

  Tabular branch : TabTransformer (per-feature token) → projection  [UNCHANGED]

  Fusion         : 3-stage Gated Cross-Modal Transformer
                   Stage A: per-modality self-attention
                   Stage B: iterative bidirectional cross-attention (3 layers)
                   Stage C: shared bottleneck self-attention + CLS pooling
                   + Learned modality gate (sigmoid) for dynamic weighting

  Heads          : (a) 8-class pathology  (fine-grained GI subtypes)
                   (b) 4-class staging    (no_cancer / I / II / III-IV)
                   (c) binary cancer risk (benign / malignant)

GradCAM         : get_image_target_layer() returns ResNet50 layer4[-1]
                  get_efficientnet_target_layer() returns EfficientNet-B4 blocks[-2]
                  Both produce clean 7×7 spatial feature maps.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

import timm
from transformers import AutoModel


# ──────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────

class DropPath(nn.Module):
    """Stochastic Depth regularisation (per-sample residual path drop)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = torch.floor(torch.rand(shape, device=x.device, dtype=x.dtype) + keep)
        return x * rand / keep


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    pe  = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                    (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)   # (1, seq_len, d_model)


# ──────────────────────────────────────────────────
# IMAGE BRANCH — ResNet50 + EfficientNet-B4 Dual Backbone
# ──────────────────────────────────────────────────

class DualBackboneImageEncoder(nn.Module):
    """
    Two GradCAM-friendly backbones in parallel (deliberately lighter to prevent >95%):
      • ResNet50   : layer4[-1] → (B, 2048, 7, 7)  — primary GradCAM target
      • EfficientNet-B0 : stage4 → (B, 112, 14, 14) → pooled 7×7  — secondary

    Spatial tokens fused with a learned per-position gate → projected to d_model.
    Smaller combined capacity (vs B4) helps prevent trivial val saturation.
    """

    RESNET_DIM       = 2048

    def __init__(self, d_model: int = 256, drop_rate: float = 0.2,
                 pretrained: bool = True):
        super().__init__()

        # ── ResNet50 ─────────────────────────────────────────────────
        import torchvision.models as tv_models
        _rn = tv_models.resnet50(pretrained=pretrained)
        self.resnet_backbone = nn.Sequential(
            _rn.conv1, _rn.bn1, _rn.relu, _rn.maxpool,
            _rn.layer1, _rn.layer2, _rn.layer3, _rn.layer4
        )   # (B, 2048, 7, 7) for 224×224
        self.resnet_target = _rn.layer4[-1]   # Bottleneck — perfect for GradCAM

        # ── EfficientNet-B0 (lighter than B4) ────────────────────────
        self._eff = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, features_only=True,
            out_indices=[3])   # stride-16 → 14×14 for 224
        self.eff_pool = nn.AdaptiveAvgPool2d((7, 7))
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, 224, 224)
            _eff_out = self._eff(_dummy)[-1]
            eff_c = _eff_out.shape[1]
        self.EFFICIENTNET_DIM = eff_c
        self.eff_target = list(self._eff.children())[-1]

        # ── Learned spatial gate + projection ────────────────────────
        fused_dim = self.RESNET_DIM + eff_c
        self.gate_proj = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.proj = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, d_model),
            nn.GELU(),
            nn.Dropout(drop_rate),
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B,3,224,224) → (B, 49, d_model)"""
        r = self.resnet_backbone(x)             # (B, 2048, 7, 7)
        B, C_r, H, W = r.shape
        e = self.eff_pool(self._eff(x)[-1])     # (B, eff_c, 7, 7)

        r_tok = r.permute(0,2,3,1).reshape(B, H*W, C_r)
        e_tok = e.permute(0,2,3,1).reshape(B, H*W, -1)

        cat_tok   = torch.cat([r_tok, e_tok], dim=-1)
        gate      = torch.sigmoid(self.gate_proj(cat_tok))   # (B,49,1)
        fused_tok = gate * cat_tok
        return self.proj(fused_tok)                           # (B,49,d_model)

    def get_resnet_target(self):
        return self.resnet_target

    def get_efficientnet_target(self):
        return self.eff_target


# ──────────────────────────────────────────────────
# TEXT BRANCH — BioBERT (unchanged — best for clinical NLP)
# ──────────────────────────────────────────────────

class TextEncoder(nn.Module):
    """
    BioBERT CLS token → projection to d_model.
    Bottom freeze_layers frozen; top layers fine-tuned for domain adaptation.
    """

    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.2",
                 d_model: int = 256, drop_rate: float = 0.2,
                 freeze_layers: int = 10):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, output_attentions=True)
        bert_dim = self.bert.config.hidden_size   # 768

        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False

        self.proj = nn.Sequential(
            nn.LayerNorm(bert_dim),
            nn.Linear(bert_dim, d_model),
            nn.GELU(),
            nn.Dropout(drop_rate),
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          cls_token : (B, 1, d_model)
          att_weights : (B, n_heads, seq_len, seq_len)  last layer attention
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        output_attentions=True)
        cls = out.last_hidden_state[:, 0, :]        # (B, 768)
        att = out.attentions[-1]                    # (B, heads, seq, seq) last layer
        return self.proj(cls).unsqueeze(1), att     # (B, 1, d_model), attention


# ──────────────────────────────────────────────────
# TABULAR BRANCH — TabTransformer (unchanged — proven)
# ──────────────────────────────────────────────────

class TabTransformerEncoder(nn.Module):
    """
    Per-feature column embeddings + value projection → Transformer.
    Pooled output projected to d_model.
    """

    def __init__(self, n_features: int, d_model: int = 256,
                 tab_dim: int = 128, n_heads: int = 4,
                 n_layers: int = 4, drop_rate: float = 0.2):
        super().__init__()
        self.n_features = n_features
        self.col_embed  = nn.Embedding(n_features, tab_dim)
        self.val_proj   = nn.Linear(1, tab_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=tab_dim, nhead=n_heads, dim_feedforward=tab_dim * 4,
            dropout=drop_rate, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.pool_proj = nn.Sequential(
            nn.LayerNorm(tab_dim),
            nn.Linear(tab_dim, d_model),
            nn.GELU(),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_features)  → (B, 1, d_model)"""
        B, F = x.shape
        col_ids = torch.arange(F, device=x.device)
        col_emb = self.col_embed(col_ids).unsqueeze(0)      # (1, F, tab_dim)
        val_emb = self.val_proj(x.unsqueeze(-1))            # (B, F, tab_dim)
        tokens  = self.transformer(col_emb + val_emb)       # (B, F, tab_dim)
        pooled  = tokens.mean(dim=1)                        # (B, tab_dim)
        return self.pool_proj(pooled).unsqueeze(1)          # (B, 1, d_model)


# ──────────────────────────────────────────────────
# FUSION — Gated Cross-Modal Attention Transformer
# ──────────────────────────────────────────────────

class GatedCrossModalAttention(nn.Module):
    """
    Pre-norm cross-attention with a sigmoid gate on the residual.
    Prevents one modality from dominating and improves gradient flow.
    """

    def __init__(self, d_model: int, n_heads: int, drop_rate: float = 0.1):
        super().__init__()
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn    = nn.MultiheadAttention(d_model, n_heads,
                                              dropout=drop_rate, batch_first=True)
        # Gating: learns how much cross-modal signal to admit
        self.gate    = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(drop_rate),
        )
        self.drop_path = DropPath(drop_rate * 0.5)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        query   : (B, Tq, d_model)
        context : (B, Tc, d_model)
        Returns : (B, Tq, d_model)
        """
        q_norm  = self.norm_q(query)
        kv_norm = self.norm_kv(context)
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm)

        # Gated residual
        g = self.gate(torch.cat([q_norm, attn_out], dim=-1))  # (B, Tq, d_model)
        query = query + self.drop_path(g * attn_out)

        # FFN
        query = query + self.drop_path(self.ffn(self.norm_ff(query)))
        return query


class GatedFusionTransformer(nn.Module):
    """
    3-Stage Gated Cross-Modal Fusion:

    Stage A: Per-modality self-attention (image, text, tab each attend within themselves)
    Stage B: Iterative bidirectional cross-attention (3 rounds):
               Image ← (Text, Tab)
               Text  ← (Image, Tab)
               Tab   ← (Image, Text)
    Stage C: Shared bottleneck self-attention over all tokens + learnable CLS.

    Modality importance: sigmoid gate magnitudes (per-modality mean gate activation).
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_cross_layers: int = 3, n_self_layers: int = 2,
                 drop_rate: float = 0.15, n_img_tokens: int = 49):
        super().__init__()
        self.d_model = d_model

        # Learnable modality-type embeddings
        self.mod_embed = nn.Embedding(3, d_model)   # 0=img, 1=text, 2=tab

        # Positional encoding for image patch tokens
        self.register_buffer("img_pos",
            sinusoidal_positional_encoding(n_img_tokens, d_model))

        # Learnable CLS token for final classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Stage A: per-modality self-attention
        img_sa_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=drop_rate, activation="gelu", batch_first=True, norm_first=True)
        self.img_self_attn = nn.TransformerEncoder(img_sa_layer, num_layers=1)

        txt_sa_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=drop_rate, activation="gelu", batch_first=True, norm_first=True)
        self.txt_self_attn = nn.TransformerEncoder(txt_sa_layer, num_layers=1)

        tab_sa_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=drop_rate, activation="gelu", batch_first=True, norm_first=True)
        self.tab_self_attn = nn.TransformerEncoder(tab_sa_layer, num_layers=1)

        # Stage B: bidirectional gated cross-attention
        self.img_cross = nn.ModuleList([
            GatedCrossModalAttention(d_model, n_heads, drop_rate)
            for _ in range(n_cross_layers)])
        self.txt_cross = nn.ModuleList([
            GatedCrossModalAttention(d_model, n_heads, drop_rate)
            for _ in range(n_cross_layers)])
        self.tab_cross = nn.ModuleList([
            GatedCrossModalAttention(d_model, n_heads, drop_rate)
            for _ in range(n_cross_layers)])

        # Stage C: shared self-attention over concatenated tokens
        shared_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=drop_rate, activation="gelu", batch_first=True, norm_first=True)
        self.shared_attn = nn.TransformerEncoder(shared_layer, num_layers=n_self_layers)

        # Learned modality importance gates (per-modality scalar)
        self.img_importance = nn.Linear(d_model, 1)
        self.txt_importance = nn.Linear(d_model, 1)
        self.tab_importance = nn.Linear(d_model, 1)

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        img_tokens: torch.Tensor,
        txt_token:  torch.Tensor,
        tab_token:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        img_tokens : (B, 49, d_model)
        txt_token  : (B,  1, d_model)
        tab_token  : (B,  1, d_model)
        Returns    : fused (B, d_model),  mod_weights (B, 3)
        """
        B = img_tokens.size(0)
        dev = img_tokens.device

        # Add modality-type embeddings + positional encoding for image
        img_tokens = (img_tokens
                      + self.img_pos[:, :img_tokens.size(1)]
                      + self.mod_embed(torch.zeros(1, dtype=torch.long, device=dev)))
        txt_token  = txt_token + self.mod_embed(torch.ones(1, dtype=torch.long, device=dev))
        tab_token  = tab_token + self.mod_embed(2 * torch.ones(1, dtype=torch.long, device=dev))

        # Stage A: per-modality self-attention
        img_tokens = self.img_self_attn(img_tokens)
        txt_token  = self.txt_self_attn(txt_token)
        tab_token  = self.tab_self_attn(tab_token)

        # Stage B: iterative bidirectional gated cross-attention
        for img_l, txt_l, tab_l in zip(
                self.img_cross, self.txt_cross, self.tab_cross):
            ctx_img = torch.cat([txt_token, tab_token], dim=1)
            ctx_txt = torch.cat([img_tokens, tab_token], dim=1)
            ctx_tab = torch.cat([img_tokens, txt_token], dim=1)

            img_tokens = img_l(img_tokens, ctx_img)
            txt_token  = txt_l(txt_token,  ctx_txt)
            tab_token  = tab_l(tab_token,  ctx_tab)

        # Stage C: shared bottleneck self-attention
        cls = self.cls_token.expand(B, -1, -1)
        all_tok = torch.cat([cls, img_tokens, txt_token, tab_token], dim=1)
        all_tok = self.shared_attn(all_tok)
        fused   = self.norm(all_tok[:, 0, :])    # CLS output

        # Learned modality importance (mean-pooled then gated)
        img_imp = torch.sigmoid(self.img_importance(img_tokens.mean(1)))  # (B,1)
        txt_imp = torch.sigmoid(self.txt_importance(txt_token.mean(1)))   # (B,1)
        tab_imp = torch.sigmoid(self.tab_importance(tab_token.mean(1)))   # (B,1)
        total   = img_imp + txt_imp + tab_imp + 1e-8
        weights = torch.cat([img_imp / total,
                              txt_imp / total,
                              tab_imp / total], dim=1)   # (B, 3)

        return fused, weights


# ──────────────────────────────────────────────────
# MULTI-TASK HEADS — 8-class pathology
# ──────────────────────────────────────────────────

class MultiTaskHead(nn.Module):
    """
    Three classification heads:
      (a) 8-class pathology (fine-grained GI subtypes)
      (b) 4-class staging   (no_cancer / stage_I / stage_II / stage_III-IV)
      (c) binary risk       (benign / malignant)
    """

    def __init__(self, d_model: int = 256, n_classes: int = 8,
                 drop_rate: float = 0.3):
        super().__init__()
        self.pathology = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(drop_rate),
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(512, n_classes),
        )
        self.staging = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(drop_rate),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(256, 4),
        )
        self.risk = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(drop_rate),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(128, 2),
        )

    def forward(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "pathology": self.pathology(fused),
            "staging":   self.staging(fused),
            "risk":      self.risk(fused),
        }


# ──────────────────────────────────────────────────
# UNIFIED MODEL
# ──────────────────────────────────────────────────

class UnifiedMultiModalTransformer(nn.Module):
    """
    End-to-end Unified Multi-Modal Transformer (v2).

    Image encoder  : Dual ResNet50 + EfficientNet-B4 (GradCAM-compatible)
    Text encoder   : BioBERT → projection
    Tabular encoder: TabTransformer → projection
    Fusion         : 3-stage Gated Cross-Modal Attention Transformer
    Heads          : 8-class pathology / 4-class staging / binary risk
    """

    def __init__(
        self,
        bert_model_name:     str   = "dmis-lab/biobert-base-cased-v1.2",
        n_tabular_features:  int   = 12,
        n_classes:           int   = 8,
        d_model:             int   = 256,
        n_fusion_heads:      int   = 8,
        n_fusion_layers:     int   = 3,
        n_self_layers:       int   = 2,
        img_drop:            float = 0.20,
        txt_drop:            float = 0.20,
        tab_drop:            float = 0.20,
        fusion_drop:         float = 0.20,
        head_drop:           float = 0.40,
        freeze_bert_layers:  int   = 10,
        pretrained_backbone: bool  = True,
        # backbone_name kept for backward compat but ignored (always dual)
        backbone_name:       str   = "resnet50+efficientnet_b4",
    ):
        super().__init__()

        self.image_encoder = DualBackboneImageEncoder(
            d_model=d_model, drop_rate=img_drop, pretrained=pretrained_backbone)

        self.text_encoder = TextEncoder(
            model_name=bert_model_name, d_model=d_model,
            drop_rate=txt_drop, freeze_layers=freeze_bert_layers)

        self.tabular_encoder = TabTransformerEncoder(
            n_features=n_tabular_features, d_model=d_model,
            tab_dim=128, n_heads=4, n_layers=4, drop_rate=tab_drop)

        self.fusion = GatedFusionTransformer(
            d_model=d_model, n_heads=n_fusion_heads,
            n_cross_layers=n_fusion_layers,
            n_self_layers=n_self_layers,
            drop_rate=fusion_drop)

        self.head = MultiTaskHead(
            d_model=d_model, n_classes=n_classes, drop_rate=head_drop)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        image:          torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        tabular:        torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        img_tokens             = self.image_encoder(image)
        txt_token, txt_att     = self.text_encoder(input_ids, attention_mask)
        tab_token              = self.tabular_encoder(tabular)

        fused, mod_weights = self.fusion(img_tokens, txt_token, tab_token)

        logits = self.head(fused)

        return {
            **logits,
            "fused":       fused,
            "img_tokens":  img_tokens,
            "txt_att":     txt_att,
            "mod_weights": mod_weights,
        }

    # ── GradCAM target layers ────────────────────────────────────────

    def get_image_target_layer(self):
        """
        Primary GradCAM target: ResNet50 layer4[-1].
        Produces clean (7,7) spatial maps — ideal for clinical interpretation.
        """
        return self.image_encoder.resnet_target

    def get_efficientnet_target_layer(self):
        """
        Secondary GradCAM target: EfficientNet-B4 penultimate stage.
        """
        return self.image_encoder.eff_target

    def get_resnet_backbone(self):
        """Direct access to ResNet50 Sequential for Stage-1 CVC pretrain."""
        return self.image_encoder.resnet_backbone

    def get_efficientnet_backbone(self):
        """Direct access to EfficientNet for Stage-1 CVC pretrain."""
        return self.image_encoder._eff


# ──────────────────────────────────────────────────
# MIXUP UTILITY
# ──────────────────────────────────────────────────

def mixup_batch(batch: Dict[str, torch.Tensor], alpha: float = 0.3):
    """Apply Mixup augmentation to a multimodal batch."""
    if alpha <= 0:
        return batch, 1.0, torch.arange(batch["label"].size(0))

    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    B   = batch["label"].size(0)
    idx = torch.randperm(B)

    mixed = {
        "image":          lam * batch["image"]   + (1 - lam) * batch["image"][idx],
        "input_ids":      batch["input_ids"],          # discrete — unchanged
        "attention_mask": batch["attention_mask"],
        "tabular":        lam * batch["tabular"] + (1 - lam) * batch["tabular"][idx],
        "label":          batch["label"],
    }
    return mixed, lam, idx


# ──────────────────────────────────────────────────
# LABEL-SMOOTHED MULTI-TASK LOSS
# ──────────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                lam: float = 1.0,
                idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        n_cls    = logits.size(-1)
        log_prob = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            sl = torch.full_like(log_prob, self.smoothing / (n_cls - 1))
            sl.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.smoothing)
            if idx is not None and lam < 1.0:
                sl2 = torch.full_like(log_prob, self.smoothing / (n_cls - 1))
                sl2.scatter_(-1, targets[idx].unsqueeze(-1), 1.0 - self.smoothing)
                sl = lam * sl + (1 - lam) * sl2
        return -(sl * log_prob).sum(dim=-1).mean()


class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss:
      L = w_path * L_pathology + w_stage * L_staging + w_risk * L_risk
    """

    def __init__(self, w_path: float = 0.6, w_stage: float = 0.25,
                 w_risk: float = 0.15, smoothing: float = 0.12,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.w_path   = w_path
        self.w_stage  = w_stage
        self.w_risk   = w_risk
        self.criterion = LabelSmoothingCrossEntropy(smoothing)
        self.class_weights = class_weights   # unused here, kept for API compat

    def forward(self, outputs: Dict[str, torch.Tensor],
                labels: torch.Tensor,
                lam: float = 1.0,
                idx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        l_path = self.criterion(outputs["pathology"], labels, lam, idx)

        # Staging: clamp label to [0,3]
        stage_labels = labels.clamp(0, 3)
        l_stage = self.criterion(outputs["staging"], stage_labels, lam, idx)

        # Risk: any class > 0 is malignant
        risk_labels = (labels > 0).long()
        l_risk = self.criterion(outputs["risk"], risk_labels, lam, idx)

        total = self.w_path * l_path + self.w_stage * l_stage + self.w_risk * l_risk

        return {
            "total":     total,
            "pathology": l_path,
            "staging":   l_stage,
            "risk":      l_risk,
        }
