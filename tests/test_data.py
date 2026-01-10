"""Tests for data loading utilities."""

import pytest
from unittest.mock import MagicMock, patch


class TestDataLoader:
    """Tests for data loading functions."""

    def test_load_dataset_import(self):
        from src.data.loader import load_dataset
        assert callable(load_dataset)

    def test_create_dataloader_import(self):
        from src.data.loader import create_dataloader
        assert callable(create_dataloader)


class TestPreprocessing:
    """Tests for preprocessing functions."""

    def test_preprocess_import(self):
        from src.data.preprocessing import preprocess_dataset
        assert callable(preprocess_dataset)

    def test_pack_sequences_import(self):
        from src.data.preprocessing import pack_sequences
        assert callable(pack_sequences)


class TestTokenizer:
    """Tests for tokenizer utilities."""

    def test_create_tokenizer_import(self):
        from src.data.tokenizer import create_tokenizer
        assert callable(create_tokenizer)
