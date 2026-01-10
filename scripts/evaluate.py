#!/usr/bin/env python3
"""Script for model assessment."""

import argparse
import json
from pathlib import Path

import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.models import load_model, load_tokenizer
from src.data import create_dataloader, load_dataset
from src.evaluation import compute_perplexity, run_benchmarks, run_downstream_tasks


def parse_args():
    parser = argparse.ArgumentParser(description="Assess a language model")

    parser.add_argument(
        "--model",
        type=str,
        default="facebook/MobileLLM-350M",
        help="Model name or path",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["perplexity", "benchmarks"],
        help="Tasks to run",
    )
    parser.add_argument(
        "--downstream-tasks",
        type=str,
        nargs="+",
        default=None,
        help="Downstream tasks for assessment",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/assessment.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Max samples for downstream tasks",
    )

    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model)

    device = next(model.parameters()).device
    print(f"Model loaded on: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    results = {"model": args.model}

    if "perplexity" in args.tasks:
        print("\nComputing perplexity on wikitext...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        dataloader = create_dataloader(
            dataset, tokenizer, batch_size=8, max_seq_length=1024, shuffle=False
        )
        ppl_results = compute_perplexity(model, dataloader, max_batches=100)
        results["perplexity"] = ppl_results
        print(f"Perplexity: {ppl_results['perplexity']:.2f}")

    if "benchmarks" in args.tasks:
        print("\nRunning benchmarks...")
        bench_results = run_benchmarks(model, tokenizer)
        results["benchmarks"] = bench_results
        print(f"Tokens/sec: {bench_results.get('inference_speed', {}).get('tokens_per_second', 'N/A'):.1f}")

    if "downstream" in args.tasks:
        print("\nRunning downstream tasks...")
        downstream_results = run_downstream_tasks(
            model, tokenizer, tasks=args.downstream_tasks, num_samples=args.num_samples
        )
        results["downstream"] = downstream_results

        for task, task_results in downstream_results.items():
            acc = task_results.get("accuracy", 0) * 100
            print(f"  {task}: {acc:.1f}%")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
