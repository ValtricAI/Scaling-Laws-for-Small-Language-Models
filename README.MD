# Scaling Laws for Small Language Models

Exploring how scaling laws behave in the sub-billion parameter regime, using Facebook's MobileLLM family as a case study.

## The Question

The big labs have mapped scaling laws extensively for massive models. But what happens at the small end, where models actually need to run on phones and edge devices? That's less explored territory.

MobileLLM caught my attention because of its "deep and thin" design philosophy: more layers with smaller hidden dimensions, plus architectural choices like Grouped Query Attention and SwiGLU. It's not just a shrunk large model. It's purpose-built for the sub-1B regime.

## Results

| Model | Parameters | Perplexity | Tokens/sec | Memory |
|-------|------------|------------|------------|--------|
| MobileLLM-125M | 125M | 11.96 | 23.7 | 0.7 GB |
| MobileLLM-350M | 345M | 8.98 | 21.7 | 1.2 GB |
| MobileLLM-600M | 603M | 7.98 | 17.5 | 1.7 GB |
| MobileLLM-1B | 1,005M | 7.17 | 13.3 | 2.6 GB |

Fitting a power law gives:

```
PPL = 1152 × N^(-0.246)
R² = 0.9935
```

That exponent (-0.246) is roughly 3× steeper than what Kaplan et al. (-0.076) and Hoffmann et al. (-0.089) found for large models. Interesting, but I want to be careful about what this actually means.

## What I Can and Can't Claim

**The honest limitation:** I'm fitting a power law to 4 data points. The R² looks great, but with n=4 you can fit almost anything. The confidence interval on that exponent is wide enough that I can't statistically distinguish it from the standard scaling laws.

**The comparison problem:** Kaplan and Hoffmann derived their laws across diverse architectures and training regimes. I'm measuring within a single architecture family where Facebook's team explicitly optimized each size point. The steeper exponent might just mean "MobileLLM's designers did good work tuning each variant," not "sub-1B models have fundamentally different scaling properties."

To make strong causal claims, you'd need multiple architecture families at each size, controlled training compute, and then compare exponents across families. I didn't do that.

**What I can defend:**

The empirical observations stand on their own. The 350M model hits a practical sweet spot: 25% better perplexity than 125M for only 8% speed loss. That's a real measurement, not an inferred law.

| Transition | Params Added | PPL Improvement | PPL per 100M params |
|------------|--------------|-----------------|---------------------|
| 125M → 350M | +220M | -2.98 | 1.35 |
| 350M → 600M | +258M | -0.98 | 0.38 |
| 600M → 1B | +402M | -0.81 | 0.20 |

The first 220M parameters buy 3× more perplexity improvement than the last 400M. There's a knee around 300-400M where diminishing returns kick in hard.

## The Takeaway

MobileLLM's "deep and thin" architecture seems to scale gracefully in the sub-1B regime. Whether that's a general property of small models or specific to this architecture family, I can't say from this data alone.

For practitioners deploying on edge devices: don't just shrink a large model. Purpose-built architectures like MobileLLM exploit a different part of the design space. The 350M variant in particular offers a compelling quality/efficiency tradeoff.

For researchers: there's room here for more rigorous study across multiple architecture families. The sub-1B regime is understudied relative to its practical importance.

![Scaling Results](results/figures/scaling_results.png)

## Quick Start

**Run the evaluation notebook on Colab:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ValtricAI/Scaling-Laws-for-Small-Language-Models/blob/main/notebooks/scaling_study_fixed.ipynb)

**Local setup:**
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
├── results/           # Evaluation results and reports
│   ├── report.md      # Full analysis report
│   ├── results.csv    # Raw data
│   └── figures/       # Plots
└── tests/             # Unit tests
```

## MobileLLM Architecture

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
