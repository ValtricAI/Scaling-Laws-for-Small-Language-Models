# MobileLLM Scaling Laws Report

## Overview

This study examines scaling laws for Facebook's MobileLLM family (125M-1B parameters) using perplexity evaluation on WikiText-2. All experiments were conducted on Google Colab with a T4 GPU.

## Scaling Law

```
PPL = 1152 × N^(-0.2457)
R² = 0.9935
```

The power law explains 99.35% of variance, confirming strong predictable scaling behavior in the sub-1B regime.

**Interpretation:**
- Doubling parameters → 15.7% lower perplexity
- 8× parameters (125M → 1B) → 40% lower perplexity

## Results

| Model | Parameters | Perplexity | Tokens/sec | Memory | Eval Time |
|-------|------------|------------|------------|--------|-----------|
| MobileLLM-125M | 125M | 11.96 | 23.7 | 0.7 GB | 86s |
| MobileLLM-350M | 345M | 8.98 | 21.7 | 1.2 GB | 75s |
| MobileLLM-600M | 603M | 7.98 | 17.5 | 1.7 GB | 124s |
| MobileLLM-1B | 1,005M | 7.17 | 13.3 | 2.6 GB | 174s |

![Scaling Results](figures/scaling_results.png)

## Comparison to Known Scaling Laws

| Study | Exponent (α) | Context |
|-------|--------------|---------|
| **MobileLLM (this study)** | **-0.246** | Sub-1B, deep-thin architecture |
| Kaplan et al. (OpenAI) | -0.076 | General LLMs |
| Hoffmann (Chinchilla) | -0.089 | Compute-optimal |

MobileLLM's steeper exponent (-0.246 vs ~-0.08) suggests more efficient scaling in the small model regime. This aligns with the MobileLLM paper's claim that their "deep and thin" architecture is better suited for sub-billion parameters.

## Diminishing Returns Analysis

| Transition | Param Increase | PPL Drop | PPL Drop per 100M params |
|------------|----------------|----------|--------------------------|
| 125M → 350M | +220M | -2.98 | 1.35 |
| 350M → 600M | +258M | -0.98 | 0.38 |
| 600M → 1B | +402M | -0.81 | 0.20 |

The first 220M parameters provide 3× more perplexity improvement than the last 400M parameters.

## Speed-Quality Tradeoff

| Model | PPL | Tok/s | PPL × Speed |
|-------|-----|-------|-------------|
| 125M | 11.96 | 23.7 | 283 |
| 350M | 8.98 | 21.7 | 195 |
| 600M | 7.98 | 17.5 | 140 |
| 1B | 7.17 | 13.3 | 95 |

The 125M → 350M transition is particularly efficient: 25% better perplexity for only 8% speed loss.

## Key Findings

1. **Scaling holds strongly** in sub-1B regime (R² = 0.99)
2. **Steeper scaling** than large models — small model optimization pays off
3. **Sweet spot at 350M**: Best balance of quality gain per parameter added
4. **1B viable on edge**: 2.6GB fits mobile devices, still 13 tok/s on T4
5. **Predictable**: Can extrapolate to estimate 2B model PPL ≈ 6.1

## Practical Recommendations

| Use Case | Recommended Model |
|----------|-------------------|
| Real-time mobile (latency critical) | 125M |
| Balanced mobile deployment | 350M |
| Quality-focused edge | 600M-1B |
| Server with T4 | 1B |

## Methodology

- **Dataset**: WikiText-2 (test split, 341,469 tokens)
- **Perplexity**: Sliding window method (max_length=1024, stride=512)
- **Hardware**: NVIDIA T4 GPU (Google Colab)
- **Precision**: FP16
- **Framework**: HuggingFace Transformers

## Files

- `results.csv` - Raw evaluation data
- `scaling_fit.json` - Fitted scaling law parameters
- `figures/scaling_results.png` - Visualization plots
