"""
models/information_extraction.py
---------------------------------
Information Extraction Subsystem (IES) — §4.3

The IES performs contextual analysis using a DistilBERT backbone with
domain-aware prompting. For an input article D', it produces:
    1. A classification probability score S_I (Equation 14):
           S_I = M(D' | π_d)
       where π_d represents domain-specific prompts prepended to the input.

    2. A similarity index δ_v (Equation 15) measuring semantic overlap
       between the query article and each node's local corpus profile:
           δ_v = |Tokens(D') ∩ Tokens(P_v)| / |Tokens(D')|

Architecture:
    DistilBERT (66M params, ~250 MB/round) → [CLS] pooling →
    Linear(768, 256) → GELU → Dropout(0.1) → Linear(256, 2)

Domain-aware prompting prepends a textual cue indicating the expected
analysis task (fact-checking, sentiment, credibility) to the article
before tokenisation, improving domain transfer.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizerFast

from config import cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Domain-Aware Prompting Templates
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_PROMPTS = {
    "fact_check":   "Fact-check the following news article for accuracy: ",
    "credibility":  "Assess the credibility of this news report: ",
    "sentiment":    "Analyse the sentiment and bias of this article: ",
    "default":      "Classify whether the following article contains fake news: ",
}


def apply_domain_prompt(text: str, domain: str = "default") -> str:
    """
    Prepend a domain-specific prompt to the article text (π_d in Eq. 14).

    Args:
        text   : Preprocessed article text.
        domain : One of 'fact_check', 'credibility', 'sentiment', 'default'.

    Returns:
        Prompted text string.
    """
    prompt = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["default"])
    return prompt + text


# ─────────────────────────────────────────────────────────────────────────────
# IES Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

class InformationExtractionSubsystem(nn.Module):
    """
    Transformer-based Information Extraction Subsystem (IES).

    Forward pass:
        input_ids      : (B, L) token IDs
        attention_mask : (B, L) binary attention mask
        →
        logits         : (B, 2)  raw class logits
        probs          : (B, 2)  softmax probabilities
        pooled         : (B, 768) [CLS] pooled representation

    Args:
        backbone       : Pre-trained model name (default: distilbert-base-uncased)
        hidden_dim     : Backbone hidden dimension (768 for DistilBERT)
        num_classes    : Number of output classes (2: Real / Fake)
        classifier_dim : Intermediate linear layer size
        dropout        : Dropout probability
        freeze_base    : Whether to freeze backbone weights (for fast fine-tuning)
    """

    def __init__(
        self,
        backbone:       str  = cfg.model.BACKBONE,
        hidden_dim:     int  = cfg.model.HIDDEN_DIM,
        num_classes:    int  = cfg.model.NUM_CLASSES,
        classifier_dim: int  = cfg.model.CLASSIFIER_DIM,
        dropout:        float = cfg.model.DROPOUT,
        freeze_base:    bool  = False,
    ):
        super().__init__()
        self.backbone_name = backbone

        logger.info("Loading DistilBERT backbone: %s", backbone)
        self.bert = DistilBertModel.from_pretrained(backbone)

        if freeze_base:
            logger.info("Freezing backbone parameters.")
            for param in self.bert.parameters():
                param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, classifier_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes),
        )

        # Layer norm for pooled representation
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout    = nn.Dropout(dropout)

        logger.info(
            "IES initialised — hidden_dim=%d, classifier_dim=%d, dropout=%.2f",
            hidden_dim, classifier_dim, dropout,
        )

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            logits  : (B, 2) raw logits
            probs   : (B, 2) softmax probabilities
            pooled  : (B, hidden_dim) [CLS] representation
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # DistilBERT returns last_hidden_state: (B, L, H)
        # Use [CLS] token (index 0) as the sequence representation
        cls_output = outputs.last_hidden_state[:, 0, :]    # (B, H)
        pooled     = self.layer_norm(cls_output)
        pooled     = self.dropout(pooled)

        logits = self.classifier(pooled)                   # (B, 2)
        probs  = F.softmax(logits, dim=-1)                 # (B, 2)

        return logits, probs, pooled

    def get_attention_weights(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Extract attention weights from all transformer layers.
        Used by the explainability module for attention analysis.

        Returns:
            List of attention tensors, one per layer.
            Each tensor shape: (B, num_heads, L, L)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        return list(outputs.attentions)   # List of (B, H, L, L)

    @property
    def num_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Similarity Index Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_similarity_index(
    query_tokens: Set[str],
    node_token_profiles: Dict[int, Set[str]],
) -> Dict[int, float]:
    """
    Compute token-based similarity index δ_v for each node (Equation 15):

        δ_v = |Tokens(D') ∩ Tokens(P_v)| / |Tokens(D')|

    This measures how semantically relevant the query article is to each
    virtual node's local corpus, guiding cost-aware node selection.

    Args:
        query_tokens        : Set of tokens from the query article D'.
        node_token_profiles : Dict mapping node_id → set of characteristic tokens.

    Returns:
        Dict mapping node_id → similarity score ∈ [0, 1].
    """
    if not query_tokens:
        return {v: 0.0 for v in node_token_profiles}

    similarities = {}
    for node_id, node_tokens in node_token_profiles.items():
        intersection = query_tokens & node_tokens
        similarities[node_id] = len(intersection) / len(query_tokens)

    return similarities


def build_node_token_profiles(
    node_texts: Dict[int, List[str]],
    top_k: int = 500,
) -> Dict[int, Set[str]]:
    """
    Build token-vocabulary profiles for each virtual node from its local corpus.

    The profile is the set of the top-k most frequent tokens in the node's
    local data, used as a lightweight proxy for semantic content.

    Args:
        node_texts : Dict mapping node_id → list of article strings.
        top_k      : Number of top tokens to retain per node.

    Returns:
        Dict mapping node_id → set of characteristic tokens.
    """
    from collections import Counter

    profiles = {}
    for node_id, texts in node_texts.items():
        token_counter: Counter = Counter()
        for text in texts:
            tokens = text.lower().split()
            token_counter.update(tokens)
        top_tokens = {tok for tok, _ in token_counter.most_common(top_k)}
        profiles[node_id] = top_tokens

    return profiles
