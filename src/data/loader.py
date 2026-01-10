"""Data loading utilities for training."""

from typing import Optional, Union
from datasets import load_dataset as hf_load_dataset, Dataset, IterableDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer


def load_dataset(
    name: str = "openwebtext",
    split: str = "train",
    streaming: bool = False,
    **kwargs,
) -> Union[Dataset, IterableDataset]:
    """Load a dataset from HuggingFace.

    Args:
        name: Dataset name on HuggingFace Hub.
        split: Dataset split to load.
        streaming: Whether to stream the dataset.
        **kwargs: Additional arguments to pass to load_dataset.

    Returns:
        Loaded dataset.
    """
    return hf_load_dataset(name, split=split, streaming=streaming, **kwargs)


def create_dataloader(
    dataset: Union[Dataset, IterableDataset],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 32,
    max_seq_length: int = 2048,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader for language model training.

    Args:
        dataset: Input dataset.
        tokenizer: Tokenizer for encoding text.
        batch_size: Batch size.
        max_seq_length: Maximum sequence length.
        num_workers: Number of data loading workers.
        shuffle: Whether to shuffle the data.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        Configured DataLoader.
    """

    def collate_fn(examples):
        texts = [ex["text"] for ex in examples]
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        encodings["labels"] = encodings["input_ids"].clone()
        return encodings

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        pin_memory=True,
    )
