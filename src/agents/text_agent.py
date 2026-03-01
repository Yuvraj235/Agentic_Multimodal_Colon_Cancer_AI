"""
Text Agent
──────────
Interprets clinical text using attention rollout from BioBERT.
Produces token-level importance scores and key clinical phrases.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional


CANCER_KEYWORDS = {
    "high_risk": ["polyp", "adenoma", "carcinoma", "cancer", "malignant",
                  "ulcerative colitis grade 3", "colitis", "lesion", "biopsy"],
    "moderate_risk": ["pathological", "abnormal", "suspicious", "nodule",
                      "hemorrhoid", "polyp", "barretts"],
    "low_risk": ["normal", "healthy", "landmark", "cecum", "ileum", "benign"],
}


@dataclass
class TextEvidence:
    text: str
    cls_embedding: np.ndarray          # (bert_dim,)
    token_importance: Dict[str, float]  # token → attention weight
    key_phrases: List[str]
    risk_level: str                     # "high" | "moderate" | "low"
    risk_flags: List[str]
    confidence: float


def _attention_rollout(attentions) -> np.ndarray:
    """Aggregate all BERT layers' attention into single saliency map."""
    # attentions: tuple of (B, heads, seq, seq)
    rollout = None
    for layer_att in attentions:
        att = layer_att[0].mean(dim=0)   # (seq, seq)
        att = att + torch.eye(att.size(0), device=att.device)
        att = att / att.sum(dim=-1, keepdim=True)
        rollout = att if rollout is None else torch.matmul(att, rollout)
    if rollout is None:
        return np.array([])
    cls_att = rollout[0, 1:].cpu().numpy()   # attention from CLS to all tokens
    if cls_att.max() > 0:
        cls_att /= cls_att.max()
    return cls_att


class TextAgent:
    """Processes clinical text and extracts XAI evidence."""

    def __init__(self, model, tokenizer, device: torch.device):
        """
        model    : UnifiedMultiModalTransformer (uses its text_encoder.bert)
        tokenizer: HuggingFace tokenizer
        """
        self.model = model.to(device)
        self.bert = model.text_encoder.bert
        self.tokenizer = tokenizer
        self.device = device

    def analyse(self, text: str, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> TextEvidence:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        self.bert.eval()
        with torch.no_grad():
            out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        cls_emb = out.last_hidden_state[0, 0].cpu().numpy()

        # Attention rollout for token importance
        att_weights = _attention_rollout(out.attentions)

        # Map back to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids[0].cpu().tolist())
        non_special = [(t, w) for t, w in zip(tokens[1:], att_weights)
                       if t not in {"[PAD]", "[SEP]", "[CLS]"}]
        token_importance = {t: float(w) for t, w in non_special}

        # Extract key phrases by high attention
        sorted_tokens = sorted(non_special, key=lambda x: x[1], reverse=True)
        key_phrases = [t for t, w in sorted_tokens[:10]
                       if not t.startswith("##") and w > 0.1]

        # Risk level from keyword matching
        text_lower = text.lower()
        risk_level = "low"
        risk_flags = []
        for kw in CANCER_KEYWORDS["high_risk"]:
            if kw in text_lower:
                risk_level = "high"
                risk_flags.append(f"HIGH_RISK_KEYWORD:{kw}")
        if risk_level == "low":
            for kw in CANCER_KEYWORDS["moderate_risk"]:
                if kw in text_lower:
                    risk_level = "moderate"
                    risk_flags.append(f"MODERATE_RISK_KEYWORD:{kw}")

        # Confidence as mean of top-5 token attentions
        top5 = [w for _, w in sorted_tokens[:5]]
        confidence = float(np.mean(top5)) if top5 else 0.5

        return TextEvidence(
            text=text,
            cls_embedding=cls_emb,
            token_importance=token_importance,
            key_phrases=key_phrases,
            risk_level=risk_level,
            risk_flags=risk_flags,
            confidence=confidence,
        )
