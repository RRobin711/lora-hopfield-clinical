"""Tests for data loading and tokenization.

Tests tokenization logic and configuration without requiring network access
to HuggingFace Hub. Uses a mock dataset to verify the tokenize_dataset
function produces correct shapes, dtypes, and column structure.
"""

from __future__ import annotations

import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from src.data import DataConfig, tokenize_dataset


@pytest.fixture()
def gpt2_tokenizer() -> AutoTokenizer:
    """GPT-2 tokenizer with pad_token set (the standard workaround)."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture()
def mock_dataset() -> Dataset:
    """Small in-memory dataset mimicking DREADDIT structure."""
    return Dataset.from_dict(
        {
            "text": [
                "I feel so stressed about everything",
                "Today was a good day at work",
                "My anxiety is getting worse",
                "Just had a relaxing weekend",
            ],
            "label": [1, 0, 1, 0],
        }
    )


class TestTokenizeDataset:
    """Verify tokenization produces correct output structure."""

    def test_output_columns(
        self, mock_dataset: Dataset, gpt2_tokenizer: AutoTokenizer
    ) -> None:
        tokenized = tokenize_dataset(mock_dataset, gpt2_tokenizer, max_length=32)
        assert "input_ids" in tokenized.column_names
        assert "attention_mask" in tokenized.column_names
        assert "labels" in tokenized.column_names

    def test_original_columns_removed(
        self, mock_dataset: Dataset, gpt2_tokenizer: AutoTokenizer
    ) -> None:
        tokenized = tokenize_dataset(mock_dataset, gpt2_tokenizer, max_length=32)
        assert "text" not in tokenized.column_names

    def test_output_length_matches_input(
        self, mock_dataset: Dataset, gpt2_tokenizer: AutoTokenizer
    ) -> None:
        tokenized = tokenize_dataset(mock_dataset, gpt2_tokenizer, max_length=32)
        assert len(tokenized) == len(mock_dataset)

    def test_max_length_enforced(
        self, mock_dataset: Dataset, gpt2_tokenizer: AutoTokenizer
    ) -> None:
        max_len = 16
        tokenized = tokenize_dataset(mock_dataset, gpt2_tokenizer, max_length=max_len)
        row = tokenized[0]
        assert len(row["input_ids"]) == max_len
        assert len(row["attention_mask"]) == max_len

    def test_torch_format(
        self, mock_dataset: Dataset, gpt2_tokenizer: AutoTokenizer
    ) -> None:
        tokenized = tokenize_dataset(mock_dataset, gpt2_tokenizer, max_length=32)
        row = tokenized[0]
        assert isinstance(row["input_ids"], torch.Tensor)
        assert isinstance(row["attention_mask"], torch.Tensor)
        assert isinstance(row["labels"], torch.Tensor)

    def test_labels_preserved(
        self, mock_dataset: Dataset, gpt2_tokenizer: AutoTokenizer
    ) -> None:
        tokenized = tokenize_dataset(mock_dataset, gpt2_tokenizer, max_length=32)
        labels = [tokenized[i]["labels"].item() for i in range(len(tokenized))]
        assert labels == [1, 0, 1, 0]

    def test_padding_creates_attention_mask(
        self, mock_dataset: Dataset, gpt2_tokenizer: AutoTokenizer
    ) -> None:
        """Short texts with large max_length should have padding (mask=0)."""
        tokenized = tokenize_dataset(mock_dataset, gpt2_tokenizer, max_length=128)
        mask = tokenized[0]["attention_mask"]
        # Should have some zeros (padding) since text is short
        assert (mask == 0).any(), "Expected padding in short text with large max_length"

    def test_pad_token_id_in_padded_positions(
        self, mock_dataset: Dataset, gpt2_tokenizer: AutoTokenizer
    ) -> None:
        """Padded positions should contain the pad token id (eos for GPT-2)."""
        tokenized = tokenize_dataset(mock_dataset, gpt2_tokenizer, max_length=128)
        ids = tokenized[0]["input_ids"]
        mask = tokenized[0]["attention_mask"]
        pad_positions = (mask == 0).nonzero(as_tuple=True)[0]
        if len(pad_positions) > 0:
            pad_ids = ids[pad_positions]
            assert (pad_ids == gpt2_tokenizer.pad_token_id).all()


class TestDataConfig:
    """Verify configuration defaults."""

    def test_defaults(self) -> None:
        cfg = DataConfig()
        assert cfg.max_length == 256
        assert cfg.batch_size == 16
        assert cfg.val_fraction == 0.2
        assert cfg.seed == 42
        assert cfg.num_workers == 0
