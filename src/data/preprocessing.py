"""Dataset preprocessing utilities."""

from typing import Optional, Callable
from datasets import Dataset
from transformers import PreTrainedTokenizer


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 2048,
    text_column: str = "text",
    num_proc: int = 4,
    remove_columns: bool = True,
) -> Dataset:
    """Preprocess and tokenize a dataset.

    Args:
        dataset: Input dataset.
        tokenizer: Tokenizer for encoding.
        max_seq_length: Maximum sequence length.
        text_column: Name of the text column.
        num_proc: Number of processes for parallel processing.
        remove_columns: Whether to remove original columns.

    Returns:
        Preprocessed dataset.
    """

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_attention_mask=True,
        )

    columns_to_remove = dataset.column_names if remove_columns else None

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=columns_to_remove,
        desc="Tokenizing",
    )

    return tokenized


def pack_sequences(
    dataset: Dataset,
    max_seq_length: int = 2048,
    num_proc: int = 4,
) -> Dataset:
    """Pack multiple sequences into fixed-length chunks for efficient training.

    Args:
        dataset: Tokenized dataset with input_ids.
        max_seq_length: Target sequence length.
        num_proc: Number of processes.

    Returns:
        Dataset with packed sequences.
    """

    def pack_examples(examples):
        all_input_ids = []
        all_attention_mask = []

        buffer_ids = []
        buffer_mask = []

        for ids, mask in zip(examples["input_ids"], examples["attention_mask"]):
            buffer_ids.extend(ids)
            buffer_mask.extend(mask)

            while len(buffer_ids) >= max_seq_length:
                all_input_ids.append(buffer_ids[:max_seq_length])
                all_attention_mask.append(buffer_mask[:max_seq_length])
                buffer_ids = buffer_ids[max_seq_length:]
                buffer_mask = buffer_mask[max_seq_length:]

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_input_ids,
        }

    return dataset.map(
        pack_examples,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Packing sequences",
    )
