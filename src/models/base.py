"""Base model utilities for loading and wrapping MobileLLM models."""

from typing import Optional, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


MOBILELLM_MODELS = {
    "125m": "facebook/MobileLLM-125M",
    "350m": "facebook/MobileLLM-350M",
    "600m": "facebook/MobileLLM-600M",
    "1b": "facebook/MobileLLM-1B",
    "1.5b": "facebook/MobileLLM-1.5B",
}


def load_tokenizer(
    model_name: str = "facebook/MobileLLM-350M",
    add_special_tokens: bool = True,
) -> PreTrainedTokenizer:
    """Load tokenizer for MobileLLM models.

    Args:
        model_name: HuggingFace model name or size key (e.g., "350m").
        add_special_tokens: Whether to add special tokens if missing.

    Returns:
        Configured tokenizer.
    """
    if model_name.lower() in MOBILELLM_MODELS:
        model_name = MOBILELLM_MODELS[model_name.lower()]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if add_special_tokens and tokenizer.eos_token is None:
        tokenizer.add_special_tokens({
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
        })

    return tokenizer


def load_model(
    model_name: str = "facebook/MobileLLM-350M",
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
) -> PreTrainedModel:
    """Load a MobileLLM model.

    Args:
        model_name: HuggingFace model name or size key (e.g., "350m").
        device: Device to load model on. If None, uses auto device map.
        dtype: Data type for model weights. If None, uses auto.
        trust_remote_code: Whether to trust remote code from HuggingFace.

    Returns:
        Loaded model.
    """
    if model_name.lower() in MOBILELLM_MODELS:
        model_name = MOBILELLM_MODELS[model_name.lower()]

    load_kwargs = {"trust_remote_code": trust_remote_code}

    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    else:
        load_kwargs["torch_dtype"] = "auto"

    if device is not None:
        load_kwargs["device_map"] = device
    else:
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    return model


class MobileLLMWrapper:
    """Wrapper for MobileLLM models with utilities for scaling law experiments."""

    def __init__(
        self,
        model_name: str = "facebook/MobileLLM-350M",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model_name = model_name
        self.tokenizer = load_tokenizer(model_name)
        self.model = load_model(model_name, device=device, dtype=dtype)

    @property
    def num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.model.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def config(self):
        """Return model config."""
        return self.model.config

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            **kwargs: Additional generation arguments.

        Returns:
            Generated text.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def compute_flops_per_token(self) -> float:
        """Estimate FLOPs per token for this model.

        Uses the approximation: FLOPs ≈ 6 * N (for forward + backward pass)
        where N is the number of parameters.
        """
        return 6 * self.num_parameters

    def get_model_info(self) -> dict:
        """Return model architecture information."""
        config = self.config
        return {
            "name": self.model_name,
            "num_parameters": self.num_parameters,
            "num_layers": getattr(config, "num_hidden_layers", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "num_attention_heads": getattr(config, "num_attention_heads", None),
            "num_kv_heads": getattr(config, "num_key_value_heads", None),
            "vocab_size": getattr(config, "vocab_size", None),
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        }
