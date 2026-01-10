#!/usr/bin/env python3
"""Run scaling sweep experiments across model sizes and compute budgets."""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.models import load_model, load_tokenizer, MobileLLMWrapper
from src.evaluation import compute_perplexity, run_benchmarks
from src.scaling import ChinchillaScalingLaw, compute_optimal_allocation


MOBILELLM_MODELS = [
    "facebook/MobileLLM-125M",
    "facebook/MobileLLM-350M",
    "facebook/MobileLLM-600M",
    "facebook/MobileLLM-1B",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run scaling sweep experiments")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/scaling_sweep.yaml",
        help="Experiment config file",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Models to analyze (default: MobileLLM family)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory",
    )

    return parser.parse_args()


def analyze_model(model_name: str) -> Dict:
    """Analyze a single model."""
    print(f"\nAnalyzing: {model_name}")

    wrapper = MobileLLMWrapper(model_name)
    info = wrapper.get_model_info()

    results = {
        "model": model_name,
        "num_parameters": info["num_parameters"],
        "architecture": info,
        "flops_per_token": wrapper.compute_flops_per_token(),
    }

    print(f"  Parameters: {info['num_parameters']:,}")
    print(f"  Layers: {info['num_layers']}")
    print(f"  Hidden size: {info['hidden_size']}")

    return results


def main():
    load_dotenv()
    args = parse_args()

    config = OmegaConf.load(args.config)

    models = args.models or MOBILELLM_MODELS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Scaling Sweep Experiment")
    print("=" * 60)

    results = []
    for model_name in models:
        try:
            model_results = analyze_model(model_name)
            results.append(model_results)
        except Exception as e:
            print(f"  Error: {e}")

    # Fit scaling law
    print("\n" + "=" * 60)
    print("Scaling Law Analysis")
    print("=" * 60)

    if len(results) >= 2:
        scaling_law = ChinchillaScalingLaw()

        for budget in config.scaling.compute_budgets:
            allocation = compute_optimal_allocation(budget)
            print(f"\nCompute budget: {budget:.1e} FLOPs")
            print(f"  Optimal params: {allocation['num_parameters']:.1e}")
            print(f"  Optimal tokens: {allocation['num_tokens']:.1e}")
            print(f"  Tokens/param: {allocation['tokens_per_param']:.1f}")

    # Save results
    output_file = output_dir / "scaling_sweep_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
