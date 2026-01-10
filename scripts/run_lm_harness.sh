#!/bin/bash
# Run lm-evaluation-harness on MobileLLM models
# Usage: ./scripts/run_lm_harness.sh [model_size]
# Example: ./scripts/run_lm_harness.sh 350M

set -e

MODEL_SIZE=${1:-"350M"}
BATCH_SIZE=${2:-8}

# Map size to HuggingFace model name
case $MODEL_SIZE in
    125M|125m)
        MODEL="facebook/MobileLLM-125M"
        OUTPUT_DIR="results/mobilellm_125m"
        ;;
    350M|350m)
        MODEL="facebook/MobileLLM-350M"
        OUTPUT_DIR="results/mobilellm_350m"
        ;;
    600M|600m)
        MODEL="facebook/MobileLLM-600M"
        OUTPUT_DIR="results/mobilellm_600m"
        ;;
    1B|1b)
        MODEL="facebook/MobileLLM-1B"
        OUTPUT_DIR="results/mobilellm_1b"
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE"
        echo "Valid options: 125M, 350M, 600M, 1B"
        exit 1
        ;;
esac

# Tasks matching MobileLLM paper benchmarks
TASKS="arc_easy,arc_challenge,boolq,piqa,hellaswag,winogrande,openbookqa"

echo "=================================="
echo "Running lm-harness on $MODEL"
echo "Output: $OUTPUT_DIR"
echo "Tasks: $TASKS"
echo "Batch size: $BATCH_SIZE"
echo "=================================="

mkdir -p "$OUTPUT_DIR"

lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL,trust_remote_code=True" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_DIR" \
    --log_samples

echo "Results saved to $OUTPUT_DIR"
