"""Perplexity computation utilities."""

import time
from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm


def compute_perplexity(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: Optional[str] = None,
    max_batches: Optional[int] = None,
) -> dict:
    """Compute perplexity on a dataset."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            if max_batches and num_batches >= max_batches:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            attention_mask = batch.get("attention_mask", torch.ones_like(batch["input_ids"]))
            num_tokens = attention_mask.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            num_batches += 1

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "total_tokens": total_tokens,
        "num_batches": num_batches,
    }


def compute_perplexity_sliding_window(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_length: int = 1024,
    stride: int = 512,
    device: Optional[str] = None,
) -> Dict:
    """Compute perplexity using sliding window approach.

    This is the HuggingFace-recommended method that avoids edge artifacts
    by using overlapping windows and only computing NLL on the non-overlapping
    portion.

    Reference: https://huggingface.co/docs/transformers/perplexity

    Args:
        model: The language model.
        tokenizer: Tokenizer for the model.
        dataset_name: HuggingFace dataset name.
        dataset_config: Dataset configuration.
        split: Dataset split to use.
        max_length: Maximum context length.
        stride: Stride for sliding window (typically max_length // 2).
        device: Device to run on.

    Returns:
        Dictionary with perplexity, tokens, timing stats.
    """
    if device is None:
        device = next(model.parameters()).device

    # Load and concatenate dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")

    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    model.eval()
    start_time = time.time()

    with torch.no_grad():
        for begin_loc in tqdm(range(0, seq_len, stride), desc="Computing perplexity"):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # Only score non-overlapping portion

            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()

            # Mask tokens we've already scored
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    wall_time = time.time() - start_time
    total_nll = torch.stack(nlls).sum()
    perplexity = torch.exp(total_nll / seq_len).item()

    return {
        "perplexity": perplexity,
        "total_tokens": seq_len,
        "wall_clock_seconds": wall_time,
        "tokens_per_second": seq_len / wall_time,
    }
