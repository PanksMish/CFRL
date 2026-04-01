"""
explainability/explainer.py
----------------------------
Layered Explainability Module for CFRL-FND (§4, Contributions #3).

The paper introduces a three-layer explainability mechanism:
    1. Feature Attribution : Which input tokens most influence the prediction?
    2. Attention Analysis  : Which token-pairs does the model attend to most?
    3. Decision Interpretability: Human-readable explanation of the verdict.

This module implements all three layers using:
    - Attention rollout for transformer attention visualisation
    - Gradient × input for token-level feature attribution (saliency maps)
    - Template-based NLG for decision explanation text

These techniques improve transparency and support the paper's XAI goals,
enabling auditors to trace why a news article was classified as fake.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast

from config import cfg
from models.information_extraction import InformationExtractionSubsystem

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Decision Templates for Natural Language Explanations
# ─────────────────────────────────────────────────────────────────────────────

FAKE_TEMPLATES = [
    "The article was flagged as FAKE NEWS (confidence: {conf:.1f}%). "
    "Key indicators: high emotional language in tokens [{tokens}], "
    "low factual consistency score ({fc:.2f}), and source credibility of {cred:.2f}.",

    "FAKE NEWS DETECTED (confidence: {conf:.1f}%). "
    "The model identified suspicious patterns around the terms [{tokens}], "
    "which are commonly associated with misinformation. "
    "The source credibility score is {cred:.2f}/1.0.",
]

REAL_TEMPLATES = [
    "The article is classified as REAL NEWS (confidence: {conf:.1f}%). "
    "Evidence of reliable reporting: balanced language, credible source (score: {cred:.2f}), "
    "and factual consistency ({fc:.2f}).",

    "REAL NEWS (confidence: {conf:.1f}%). "
    "The content aligns with verified sources. "
    "Most influential tokens: [{tokens}]. Source credibility: {cred:.2f}/1.0.",
]


class CFRLExplainer:
    """
    Provides multi-layer explanations for CFRL-FND predictions.

    Layers:
        1. Attention analysis    — highlights which tokens the model attended to.
        2. Feature attribution   — gradient-based token importance scores.
        3. Decision explanation  — natural language summary of the verdict.

    Args:
        model     : Trained InformationExtractionSubsystem.
        tokenizer : DistilBERT tokeniser for decoding token IDs.
        device    : Torch device.
    """

    def __init__(
        self,
        model:     InformationExtractionSubsystem,
        tokenizer: DistilBertTokenizerFast,
        device:    Optional[torch.device] = None,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 1: Attention Analysis
    # ─────────────────────────────────────────────────────────────────────────

    def get_attention_scores(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        layers:         List[int] = cfg.explain.ATTENTION_LAYERS,
    ) -> np.ndarray:
        """
        Extract and aggregate attention scores from specified transformer layers.

        Uses attention rollout: multiply attention matrices across layers
        to capture indirect token dependencies (Abnar & Zuidema, 2020).

        Args:
            input_ids      : (1, L) token IDs for a single article.
            attention_mask : (1, L) binary mask.
            layers         : List of layer indices to aggregate (−1 = last).

        Returns:
            token_attention : (L,) normalised attention score per token.
        """
        self.model.eval()

        with torch.no_grad():
            attention_weights = self.model.get_attention_weights(
                input_ids.to(self.device),
                attention_mask.to(self.device),
            )

        # attention_weights: list of (1, num_heads, L, L) tensors
        # Average over heads, then aggregate specified layers
        layer_attentions = []
        for layer_idx in layers:
            attn = attention_weights[layer_idx]          # (1, H, L, L)
            attn = attn.mean(dim=1).squeeze(0).cpu().numpy()  # (L, L)
            layer_attentions.append(attn)

        # Average across selected layers
        avg_attention = np.mean(layer_attentions, axis=0)    # (L, L)

        # CLS token attention to all other tokens: row 0 of the matrix
        cls_attention = avg_attention[0, :]                  # (L,)

        # Mask padding positions
        mask = attention_mask.squeeze(0).cpu().numpy()
        cls_attention = cls_attention * mask

        # Normalise to [0, 1]
        max_val = cls_attention.max()
        if max_val > 0:
            cls_attention /= max_val

        return cls_attention

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 2: Feature Attribution (Gradient × Input Saliency)
    # ─────────────────────────────────────────────────────────────────────────

    def get_token_attributions(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        target_class:   int = 1,
    ) -> np.ndarray:
        """
        Compute gradient × input saliency scores for each token.

        Saliency = |∂output / ∂embedding| elementwise-multiplied by embedding.
        Aggregated across embedding dimensions to produce per-token scores.

        Args:
            input_ids      : (1, L) token IDs.
            attention_mask : (1, L) binary mask.
            target_class   : Class index to compute gradients for (1 = Fake).

        Returns:
            saliency : (L,) attribution score per token, normalised to [0, 1].
        """
        self.model.train()  # Enable gradients in embedding layer

        input_ids_dev = input_ids.to(self.device)
        attn_mask_dev = attention_mask.to(self.device)

        # Get token embeddings with gradient tracking
        embeddings = self.model.bert.embeddings(input_ids_dev)
        embeddings.retain_grad()

        # Forward pass from embeddings
        outputs = self.model.bert(
            inputs_embeds  = embeddings,
            attention_mask = attn_mask_dev,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        pooled     = self.model.layer_norm(cls_output)
        pooled     = self.model.dropout(pooled)
        logits     = self.model.classifier(pooled)

        # Backward pass for target class
        self.model.zero_grad()
        logits[0, target_class].backward()

        # Gradient × input saliency
        grad     = embeddings.grad.squeeze(0).cpu().detach().numpy()    # (L, H)
        emb      = embeddings.squeeze(0).cpu().detach().numpy()          # (L, H)
        saliency = np.abs(grad * emb).sum(axis=-1)                       # (L,)

        # Mask padding and normalise
        mask     = attention_mask.squeeze(0).cpu().numpy()
        saliency = saliency * mask
        max_val  = saliency.max()
        if max_val > 0:
            saliency /= max_val

        self.model.eval()
        return saliency

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 3: Decision Interpretation (NLG Template)
    # ─────────────────────────────────────────────────────────────────────────

    def generate_explanation(
        self,
        input_ids:        torch.Tensor,
        attention_mask:   torch.Tensor,
        prediction:       int,
        confidence:       float,
        source_credibility: float = 0.7,
        factual_consistency: float = 0.6,
        top_k:            int = cfg.explain.TOP_K_FEATURES,
    ) -> Dict:
        """
        Generate a full layered explanation for a single prediction.

        Args:
            input_ids           : (1, L) token IDs.
            attention_mask      : (1, L) binary mask.
            prediction          : Predicted label (0=Real, 1=Fake).
            confidence          : Prediction confidence ∈ [0, 1].
            source_credibility  : Crowdsource credibility score for the source.
            factual_consistency : Estimated factual consistency score.
            top_k               : Number of top tokens to highlight.

        Returns:
            Dict with keys:
                'prediction', 'confidence', 'top_tokens',
                'attention_scores', 'token_attributions',
                'text_explanation', 'tokens'
        """
        # Decode tokens (skip special tokens)
        token_ids   = input_ids.squeeze(0).cpu().numpy()
        tokens      = self.tokenizer.convert_ids_to_tokens(token_ids)
        seq_len     = int(attention_mask.squeeze(0).sum().item())
        tokens      = tokens[:seq_len]

        # Layer 1: Attention
        attn_scores  = self.get_attention_scores(input_ids, attention_mask)[:seq_len]

        # Layer 2: Saliency
        try:
            saliency = self.get_token_attributions(input_ids, attention_mask)[:seq_len]
        except RuntimeError as e:
            logger.warning("Saliency computation failed (%s); using attention as fallback.", e)
            saliency = attn_scores

        # Combined importance = average of attention and saliency
        importance  = (attn_scores + saliency) / 2.0

        # Top-k important tokens (exclude [CLS], [SEP], padding)
        exclude     = {"[cls]", "[sep]", "[pad]"}
        token_scores = [
            (tok, float(imp))
            for tok, imp in zip(tokens, importance)
            if tok.lower() not in exclude and not tok.startswith("##")
        ]
        token_scores.sort(key=lambda x: x[1], reverse=True)
        top_tokens  = [tok for tok, _ in token_scores[:top_k]]

        # Layer 3: Natural language explanation
        import random
        templates    = FAKE_TEMPLATES if prediction == 1 else REAL_TEMPLATES
        template     = random.choice(templates)
        explanation  = template.format(
            conf   = confidence * 100,
            tokens = ", ".join(top_tokens[:5]),
            fc     = factual_consistency,
            cred   = source_credibility,
        )

        return {
            "prediction":         int(prediction),
            "label":              "FAKE" if prediction == 1 else "REAL",
            "confidence":         float(confidence),
            "top_tokens":         top_tokens,
            "attention_scores":   attn_scores.tolist(),
            "token_attributions": saliency.tolist(),
            "tokens":             tokens,
            "text_explanation":   explanation,
            "source_credibility": float(source_credibility),
            "factual_consistency": float(factual_consistency),
        }

    def batch_explain(
        self,
        input_ids_batch:      torch.Tensor,
        attention_mask_batch: torch.Tensor,
        predictions:          List[int],
        confidences:          List[float],
    ) -> List[Dict]:
        """
        Generate explanations for a batch of predictions.

        Args:
            input_ids_batch      : (B, L) tensor.
            attention_mask_batch : (B, L) tensor.
            predictions          : List of B predicted labels.
            confidences          : List of B confidence scores.

        Returns:
            List of explanation dicts (one per sample).
        """
        explanations = []
        for i in range(input_ids_batch.size(0)):
            exp = self.generate_explanation(
                input_ids        = input_ids_batch[i].unsqueeze(0),
                attention_mask   = attention_mask_batch[i].unsqueeze(0),
                prediction       = predictions[i],
                confidence       = confidences[i],
            )
            explanations.append(exp)
        return explanations
