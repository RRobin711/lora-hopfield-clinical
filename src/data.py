"""
DREADDIT dataset loading and tokenization for GPT-2 classification.

DREADDIT (Turcan & McKeown, 2019) is a binary stress detection dataset from
Reddit (~3500 samples across 5 subreddits). Chosen for Phase 2 because:
- Connects to Phase 1 domain: stress is upstream of PHQ-9/GAD-7 clinical scales
- Informal Reddit language mirrors the layperson query failure mode from Phase 1's
  HyDE experiment — both involve informal text hitting a specialized model
- Small enough for the RX 6700S (8GB VRAM) with GPT-2 small (~20-40 min/run)
- Binary classification gives clean metrics for the rank ablation table

GPT-2 tokenizer note: GPT-2 ships without a pad token. We set pad_token = eos_token
(token id 50256). The attention mask zeros out pad positions so the model ignores
them, and the classification head extracts the last *real* token's hidden state,
not the last sequence position (which would be a meaningless pad token embedding).

Dataset source: andreagasparini/dreaddit on HuggingFace Hub.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase


@dataclass
class DataConfig:
    """Configuration for DREADDIT data loading and tokenization.

    Attributes:
        dataset_name: HuggingFace Hub path for DREADDIT.
        max_length: Maximum token length for GPT-2 tokenization. 256 balances
            capturing enough Reddit post context with VRAM efficiency on 8GB.
        batch_size: Training batch size. 16 fits comfortably in 8GB VRAM
            with GPT-2 small + LoRA overhead.
        val_fraction: Fraction of train split held out for validation.
            Used for early stopping; official test set stays untouched.
        seed: Random seed for train/val stratified split reproducibility.
        num_workers: DataLoader workers. 0 for HDD setup (avoids redundant
            disk reads — data is pre-tokenized in memory).
    """

    dataset_name: str = "andreagasparini/dreaddit"
    max_length: int = 256
    batch_size: int = 16
    val_fraction: float = 0.2
    seed: int = 42
    num_workers: int = 0


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    """Tokenize a HuggingFace Dataset of text+label pairs for GPT-2.

    Tokenizes all rows at once and keeps the result in memory — DREADDIT
    is ~3500 rows, so this is trivially small. Avoids per-epoch disk reads
    on the HDD setup.

    Args:
        dataset: HuggingFace Dataset with 'text' and 'label' columns.
        tokenizer: GPT-2 tokenizer with pad_token already set.
        max_length: Maximum sequence length (truncation + padding target).

    Returns:
        Dataset with columns: input_ids, attention_mask, labels.
        All tensors are in PyTorch format.
    """

    def _tokenize(examples: dict) -> dict:
        tokens = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        tokens["labels"] = examples["label"]
        return tokens

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    tokenized.set_format("torch")
    return tokenized


def load_dreaddit(
    config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load DREADDIT, tokenize, split, and return train/val/test DataLoaders.

    The official train split is further divided into train/val (stratified by
    label) for early stopping. The official test split is held out for final
    evaluation only.

    Args:
        config: DataConfig controlling tokenization and batching.
        tokenizer: GPT-2 tokenizer. Must have pad_token set before calling
            (caller's responsibility — typically tokenizer.pad_token = tokenizer.eos_token).

    Returns:
        Tuple of (train_loader, val_loader, test_loader). Each batch is a dict:
            - input_ids: (batch_size, max_length) long tensor
            - attention_mask: (batch_size, max_length) long tensor
            - labels: (batch_size,) long tensor, values in {0, 1}

    Raises:
        ValueError: If tokenizer has no pad_token set.
    """
    if tokenizer.pad_token is None:
        raise ValueError(
            "Tokenizer has no pad_token set. For GPT-2, set "
            "tokenizer.pad_token = tokenizer.eos_token before calling load_dreaddit."
        )

    raw = load_dataset(config.dataset_name)

    # Stratified train/val split from the official train set
    train_val = raw["train"].train_test_split(
        test_size=config.val_fraction,
        seed=config.seed,
        stratify_by_column="label",
    )
    train_ds = tokenize_dataset(train_val["train"], tokenizer, config.max_length)
    val_ds = tokenize_dataset(train_val["test"], tokenizer, config.max_length)
    test_ds = tokenize_dataset(raw["test"], tokenizer, config.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(config.seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
