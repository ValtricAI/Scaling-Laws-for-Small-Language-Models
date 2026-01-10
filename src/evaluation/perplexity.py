"""Perplexity computation utilities."""

from typing import Optional, Union
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
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
