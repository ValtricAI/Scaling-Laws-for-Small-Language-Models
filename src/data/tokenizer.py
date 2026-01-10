"""Tokenizer utilities."""

from transformers import AutoTokenizer, PreTrainedTokenizer


def create_tokenizer(
    model_name: str = "facebook/MobileLLM-350M",
    add_special_tokens: bool = True,
    padding_side: str = "right",
) -> PreTrainedTokenizer:
    """Create and configure a tokenizer.

    Args:
        model_name: HuggingFace model name.
        add_special_tokens: Whether to add special tokens if missing.
        padding_side: Which side to pad on ("left" or "right").

    Returns:
        Configured tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.padding_side = padding_side

    if add_special_tokens:
        special_tokens = {}
        if tokenizer.eos_token is None:
            special_tokens["eos_token"] = "</s>"
        if tokenizer.bos_token is None:
            special_tokens["bos_token"] = "<s>"
        if tokenizer.unk_token is None:
            special_tokens["unk_token"] = "<unk>"
        if tokenizer.pad_token is None:
            special_tokens["pad_token"] = tokenizer.eos_token or "</s>"

        if special_tokens:
            tokenizer.add_special_tokens(special_tokens)

    return tokenizer
