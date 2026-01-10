# Scaling Laws for Small Language Models

Research project exploring scaling laws for small language models, with a focus on Facebook's MobileLLM family.

## What's This About?

I'm investigating how scaling laws apply to sub-billion parameter models. The big labs have done extensive work on scaling laws for massive models, but there's less research on what happens at the smaller end - where models actually run on devices.

MobileLLM is particularly interesting because of its "deep and thin" architecture philosophy: more layers with smaller hidden dimensions, plus tricks like Grouped Query Attention and SwiGLU.

## Current Status

Project structure is set up. Waiting on HuggingFace license approval for MobileLLM access.

**What's implemented:**
- Scaling law implementations (Chinchilla & Kaplan formulations)
- Attention variants (GQA, Linear, Sparse)
- Training infrastructure with W&B logging
- Evaluation suite for downstream tasks

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

You'll need to authenticate with HuggingFace and accept the MobileLLM license:
```bash
huggingface-cli login
```

Then visit https://huggingface.co/facebook/MobileLLM-350M to accept the license.

## Project Structure

```
├── configs/           # Model, training, and experiment configs
├── src/
│   ├── data/          # Data loading and preprocessing
│   ├── models/        # Attention mechanisms and architectures
│   ├── training/      # Trainer, optimizer, scheduler
│   ├── evaluation/    # Benchmarks and metrics
│   └── scaling/       # Scaling law implementations
├── scripts/           # Training and evaluation scripts
├── notebooks/         # Analysis notebooks
└── tests/             # Unit tests
```

## MobileLLM Models

| Model | Params | Layers | Hidden | Heads | KV Heads |
|-------|--------|--------|--------|-------|----------|
| MobileLLM-125M | 124.6M | 30 | 576 | 9 | 3 |
| MobileLLM-350M | 345.3M | 32 | 960 | 15 | 5 |
| MobileLLM-600M | 603.1M | 40 | 1152 | 18 | 6 |
| MobileLLM-1B | 1.01B | 54 | 1280 | 20 | 5 |

## References

- [MobileLLM Paper](https://arxiv.org/abs/2402.14905) - Optimizing Sub-billion Parameter Language Models for On-Device Use Cases
- [Chinchilla Paper](https://arxiv.org/abs/2203.15556) - Training Compute-Optimal Large Language Models
- [Kaplan et al.](https://arxiv.org/abs/2001.08361) - Scaling Laws for Neural Language Models
