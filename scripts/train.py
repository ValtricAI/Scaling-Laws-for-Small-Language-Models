#!/usr/bin/env python3
"""Training script for MobileLLM scaling law experiments."""

import argparse
import os
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.models import load_model, load_tokenizer
from src.data import create_dataloader, load_dataset, preprocess_dataset
from src.training import Trainer, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/default.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model/base.yaml",
        help="Path to model config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    # Load configs
    train_config = OmegaConf.load(args.config)
    model_config = OmegaConf.load(args.model_config)

    print(f"Training config: {args.config}")
    print(f"Model config: {args.model_config}")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project or train_config.get("logging", {}).get("wandb", {}).get("project", "scaling-laws-slm"),
            config={
                "training": OmegaConf.to_container(train_config),
                "model": OmegaConf.to_container(model_config),
            },
        )

    # Load model and tokenizer
    model_name = model_config.model.name
    print(f"Loading model: {model_name}")

    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load dataset
    data_config = train_config.get("data", {})
    dataset_name = data_config.get("dataset", "openwebtext")
    max_seq_length = train_config.training.get("max_seq_length", 2048)

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train", streaming=True)

    # Create dataloader
    train_dataloader = create_dataloader(
        dataset,
        tokenizer,
        batch_size=train_config.training.batch_size,
        max_seq_length=max_seq_length,
        num_workers=data_config.get("num_workers", 4),
    )

    # Setup training config
    training_config = TrainingConfig(
        learning_rate=train_config.training.learning_rate,
        weight_decay=train_config.training.weight_decay,
        adam_beta1=train_config.training.get("adam_beta1", 0.9),
        adam_beta2=train_config.training.get("adam_beta2", 0.95),
        warmup_steps=train_config.training.warmup_steps,
        max_steps=train_config.training.max_steps,
        gradient_accumulation_steps=train_config.training.gradient_accumulation_steps,
        logging_steps=train_config.training.logging_steps,
        fp16=train_config.training.get("fp16", False),
        bf16=train_config.training.get("bf16", True),
        output_dir=str(output_dir),
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataloader=train_dataloader,
        use_wandb=use_wandb,
    )

    # Train
    print("Starting training...")
    result = trainer.train()

    print(f"Training complete. Final step: {result['final_step']}")
    print(f"Total tokens: {result['total_tokens']:,}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
