"""
dataset/data_loader.py
----------------------
Handles loading the Kaggle Fake News Detection dataset (True.csv / Fake.csv),
applies the full preprocessing pipeline described in the paper, and exposes
PyTorch Dataset / DataLoader objects consumed by the training modules.

Preprocessing pipeline (§3.1):
  1. Merge title + text into a single article string
  2. Remove HTML tags, special symbols, and extra whitespace
  3. Lowercase transformation
  4. Tokenisation via HuggingFace DistilBERT tokeniser
  5. Padding / truncation to MAX_SEQ_LEN tokens
  6. Stratified 80 / 10 / 10 train-val-test split
"""

import os
import re
import logging
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

from config import cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Text Cleaning Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _remove_html_tags(text: str) -> str:
    """Strip HTML/XML tags from raw text."""
    return re.sub(r"<[^>]+>", " ", text)


def _remove_special_symbols(text: str) -> str:
    """Keep only alphanumeric characters and basic punctuation."""
    return re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\'\-]", " ", text)


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces / newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """
    Full text cleaning pipeline.

    Steps:
        1. Remove HTML tags
        2. Remove special symbols
        3. Lowercase
        4. Normalize whitespace
    """
    text = _remove_html_tags(text)
    text = _remove_special_symbols(text)
    text = text.lower()
    text = _normalize_whitespace(text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loading and Splitting
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_dataframe(data_dir: str) -> pd.DataFrame:
    """
    Load True.csv and Fake.csv from *data_dir*, assign binary labels,
    and return a single merged DataFrame.

    Args:
        data_dir: Directory containing True.csv and Fake.csv.

    Returns:
        DataFrame with columns ['text', 'label'] where
            label = 0 → Real news
            label = 1 → Fake news
    """
    real_path = os.path.join(data_dir, cfg.data.REAL_NEWS_FILE)
    fake_path = os.path.join(data_dir, cfg.data.FAKE_NEWS_FILE)

    if not os.path.exists(real_path) or not os.path.exists(fake_path):
        logger.warning(
            "Dataset files not found at %s. Generating synthetic dataset for demonstration.",
            data_dir,
        )
        return _generate_synthetic_dataframe()

    logger.info("Loading real news from: %s", real_path)
    df_real = pd.read_csv(real_path)
    df_real["label"] = 0  # Real → 0

    logger.info("Loading fake news from: %s", fake_path)
    df_fake = pd.read_csv(fake_path)
    df_fake["label"] = 1  # Fake → 1

    # Merge title + text into a single article field
    for df in [df_real, df_fake]:
        title_col = "title" if "title" in df.columns else None
        text_col  = "text"  if "text"  in df.columns else df.columns[0]

        if title_col:
            df["article"] = df[title_col].fillna("") + " " + df[text_col].fillna("")
        else:
            df["article"] = df[text_col].fillna("")

    df = pd.concat(
        [df_real[["article", "label"]], df_fake[["article", "label"]]],
        ignore_index=True,
    )

    # Apply text cleaning
    logger.info("Applying text cleaning pipeline ...")
    df["article"] = df["article"].apply(clean_text)

    # Shuffle
    df = df.sample(frac=1, random_state=cfg.data.RANDOM_SEED).reset_index(drop=True)

    logger.info(
        "Dataset loaded: %d real, %d fake, %d total",
        (df["label"] == 0).sum(),
        (df["label"] == 1).sum(),
        len(df),
    )
    return df


def _generate_synthetic_dataframe() -> pd.DataFrame:
    """
    Generate a small synthetic dataset when the real Kaggle files are absent.
    This mirrors the class distribution reported in the paper (Table 2).
    Used for unit-testing and CI environments.
    """
    logger.info("Generating synthetic dataset (5,000 samples) for demonstration ...")
    np.random.seed(cfg.data.RANDOM_SEED)

    real_templates = [
        "the president signed the new economic bill into law yesterday",
        "scientists discover breakthrough treatment for rare genetic disorder",
        "local government announces infrastructure investment plan",
        "central bank maintains interest rates amid economic uncertainty",
        "international summit concludes with new climate agreement",
    ]
    fake_templates = [
        "shocking conspiracy revealed government hiding alien technology",
        "miracle cure doctors do not want you to know about this remedy",
        "elites planning secret agenda to control world population",
        "breaking news celebrity found involved in massive fraud scheme",
        "exclusive leaked document reveals hidden global control network",
    ]

    n_real, n_fake = 2_400, 2_600
    rng = np.random.default_rng(cfg.data.RANDOM_SEED)

    real_articles = [
        real_templates[i % len(real_templates)] + " " + " ".join(
            [str(rng.integers(1000, 9999)) for _ in range(30)]
        )
        for i in range(n_real)
    ]
    fake_articles = [
        fake_templates[i % len(fake_templates)] + " " + " ".join(
            [str(rng.integers(1000, 9999)) for _ in range(30)]
        )
        for i in range(n_fake)
    ]

    texts  = real_articles + fake_articles
    labels = [0] * n_real + [1] * n_fake

    df = pd.DataFrame({"article": texts, "label": labels})
    df = df.sample(frac=1, random_state=cfg.data.RANDOM_SEED).reset_index(drop=True)
    return df


def load_and_split_data(
    data_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load dataset and perform stratified 80/10/10 split.

    Args:
        data_dir: Directory containing True.csv and Fake.csv.

    Returns:
        Tuple (df_train, df_val, df_test)
    """
    df = load_raw_dataframe(data_dir)

    # Stratified split: 80 % train, 10 % val, 10 % test
    df_train, df_tmp = train_test_split(
        df,
        test_size=(cfg.data.VAL_RATIO + cfg.data.TEST_RATIO),
        stratify=df["label"],
        random_state=cfg.data.RANDOM_SEED,
    )
    df_val, df_test = train_test_split(
        df_tmp,
        test_size=cfg.data.TEST_RATIO / (cfg.data.VAL_RATIO + cfg.data.TEST_RATIO),
        stratify=df_tmp["label"],
        random_state=cfg.data.RANDOM_SEED,
    )

    logger.info(
        "Split sizes — Train: %d | Val: %d | Test: %d",
        len(df_train), len(df_val), len(df_test),
    )
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FakeNewsDataset(Dataset):
    """
    PyTorch Dataset wrapping tokenised fake-news articles.

    Each item is a dictionary containing:
        input_ids      : (MAX_SEQ_LEN,) token IDs
        attention_mask : (MAX_SEQ_LEN,) binary attention mask
        label          : scalar int64 tensor (0=Real, 1=Fake)

    Args:
        df        : DataFrame with columns ['article', 'label']
        tokenizer : HuggingFace tokeniser (DistilBertTokenizerFast)
        max_len   : Maximum token sequence length
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: DistilBertTokenizerFast,
        max_len: int = cfg.data.MAX_SEQ_LEN,
    ):
        self.texts     = df["article"].tolist()
        self.labels    = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text  = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       # (max_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (max_len,)
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str,
    batch_size: int = cfg.train.BATCH_SIZE,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    End-to-end helper: load data → tokenise → return DataLoaders.

    Args:
        data_dir    : Directory containing True.csv / Fake.csv.
        batch_size  : Mini-batch size for training DataLoader.
        num_workers : Number of worker processes for DataLoader.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    logger.info("Initialising DistilBERT tokeniser: %s", cfg.model.BACKBONE)
    tokenizer = DistilBertTokenizerFast.from_pretrained(cfg.model.BACKBONE)

    df_train, df_val, df_test = load_and_split_data(data_dir)

    train_dataset = FakeNewsDataset(df_train, tokenizer)
    val_dataset   = FakeNewsDataset(df_val,   tokenizer)
    test_dataset  = FakeNewsDataset(df_test,  tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        "DataLoaders ready — Train batches: %d | Val batches: %d | Test batches: %d",
        len(train_loader), len(val_loader), len(test_loader),
    )
    return train_loader, val_loader, test_loader
