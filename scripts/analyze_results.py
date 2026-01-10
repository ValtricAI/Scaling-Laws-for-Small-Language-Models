#!/usr/bin/env python3
"""Analyze experiment results and generate visualizations."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze scaling experiment results")

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing result files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures",
        help="Output directory for figures",
    )

    return parser.parse_args()


def plot_scaling_curve(results: list, output_path: Path):
    """Plot loss vs compute scaling curve."""
    params = [r["num_parameters"] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(params, range(len(params)), s=100)
    
    for i, r in enumerate(results):
        ax.annotate(
            r["model"].split("/")[-1],
            (params[i], i),
            xytext=(10, 0),
            textcoords="offset points",
        )
    
    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Model Index")
    ax.set_title("MobileLLM Model Family")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "scaling_curve.png", dpi=150)
    plt.close()


def plot_architecture_comparison(results: list, output_path: Path):
    """Compare architectural choices across models."""
    models = [r["model"].split("/")[-1] for r in results]
    params = [r["num_parameters"] / 1e6 for r in results]
    layers = [r["architecture"]["num_layers"] for r in results]
    hidden = [r["architecture"]["hidden_size"] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Parameters
    axes[0].barh(models, params, color="steelblue")
    axes[0].set_xlabel("Parameters (M)")
    axes[0].set_title("Model Size")
    
    # Layers
    axes[1].barh(models, layers, color="darkorange")
    axes[1].set_xlabel("Number of Layers")
    axes[1].set_title("Depth")
    
    # Hidden size
    axes[2].barh(models, hidden, color="forestgreen")
    axes[2].set_xlabel("Hidden Size")
    axes[2].set_title("Width")
    
    plt.tight_layout()
    plt.savefig(output_path / "architecture_comparison.png", dpi=150)
    plt.close()


def main():
    args = parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results_file = results_dir / "scaling_sweep_results.json"
    
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        
        print(f"Loaded {len(results)} model results")
        
        # Generate plots
        plot_scaling_curve(results, output_dir)
        print(f"Saved: scaling_curve.png")
        
        plot_architecture_comparison(results, output_dir)
        print(f"Saved: architecture_comparison.png")
        
        # Summary statistics
        print("\nModel Summary:")
        print("-" * 60)
        for r in results:
            name = r["model"].split("/")[-1]
            params = r["num_parameters"] / 1e6
            layers = r["architecture"]["num_layers"]
            hidden = r["architecture"]["hidden_size"]
            print(f"{name:20s} | {params:6.1f}M | {layers:2d} layers | {hidden:4d} hidden")
    else:
        print(f"No results found at: {results_file}")
        print("Run scaling sweep first: python scripts/run_scaling_sweep.py")


if __name__ == "__main__":
    main()
