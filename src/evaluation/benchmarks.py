"""Benchmark utilities."""

from typing import Dict, List, Optional, Any
import time
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def run_benchmarks(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    benchmarks: List[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a suite of benchmarks on a model."""
    if benchmarks is None:
        benchmarks = ["inference_speed", "memory_usage"]

    if device is None:
        device = next(model.parameters()).device

    results = {}

    if "inference_speed" in benchmarks:
        results["inference_speed"] = benchmark_inference_speed(model, tokenizer, device)

    if "memory_usage" in benchmarks:
        results["memory_usage"] = benchmark_memory_usage(model, device)

    return results


def benchmark_inference_speed(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str,
    num_tokens: int = 100,
    num_runs: int = 10,
    warmup_runs: int = 3,
) -> Dict[str, float]:
    """Benchmark inference speed (tokens per second)."""
    model.eval()

    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            model.generate(
                **inputs,
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    if device == "cuda" or str(device).startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model.generate(
                **inputs,
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            if device == "cuda" or str(device).startswith("cuda"):
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    avg_time = sum(times) / len(times)
    tokens_per_second = num_tokens / avg_time

    return {
        "tokens_per_second": tokens_per_second,
        "avg_time_seconds": avg_time,
    }


def benchmark_memory_usage(model: PreTrainedModel, device: str) -> Dict[str, float]:
    """Benchmark memory usage."""
    results = {
        "model_params_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6,
    }

    if device == "cuda" or str(device).startswith("cuda"):
        results["cuda_allocated_mb"] = torch.cuda.memory_allocated() / 1e6
        results["cuda_reserved_mb"] = torch.cuda.memory_reserved() / 1e6

    return results
