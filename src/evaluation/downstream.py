"""Downstream task utilities."""

from typing import Dict, List, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm


SUPPORTED_TASKS = [
    "arc_easy", "arc_challenge", "boolq", "piqa",
    "siqa", "hellaswag", "obqa", "winogrande",
]


def run_downstream_tasks(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    tasks: Optional[List[str]] = None,
    device: Optional[str] = None,
    num_samples: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Run downstream task assessments."""
    if tasks is None:
        tasks = SUPPORTED_TASKS

    if device is None:
        device = next(model.parameters()).device

    results = {}
    for task in tasks:
        if task not in SUPPORTED_TASKS:
            continue
        print(f"Running {task}...")
        results[task] = run_single_task(model, tokenizer, task, device, num_samples)

    return results


def run_single_task(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: str,
    device: str,
    num_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Run a single downstream task."""
    dataset = load_task_dataset(task)

    if num_samples and len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for example in tqdm(dataset, desc=task):
            prompt, choices, label = format_task_example(task, example)

            choice_scores = []
            for choice in choices:
                full_text = prompt + choice
                inputs = tokenizer(full_text, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, labels=inputs["input_ids"])
                choice_scores.append(-outputs.loss.item())

            prediction = choice_scores.index(max(choice_scores))
            if prediction == label:
                correct += 1
            total += 1

    return {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total}


def load_task_dataset(task: str):
    """Load the dataset for a task."""
    configs = {
        "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
        "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
        "boolq": ("google/boolq", None, "validation"),
        "piqa": ("ybisk/piqa", None, "validation"),
        "siqa": ("allenai/social_i_qa", None, "validation"),
        "hellaswag": ("Rowan/hellaswag", None, "validation"),
        "obqa": ("allenai/openbookqa", "main", "test"),
        "winogrande": ("allenai/winogrande", "winogrande_xl", "validation"),
    }
    name, config, split = configs[task]
    return load_dataset(name, config, split=split) if config else load_dataset(name, split=split)


def format_task_example(task: str, example: dict) -> tuple:
    """Format an example for a specific task."""
    if task in ["arc_easy", "arc_challenge"]:
        prompt = f"Question: {example['question']}\nAnswer:"
        choices = example["choices"]["text"]
        label = example["choices"]["label"].index(example["answerKey"])
        return prompt, choices, label

    elif task == "boolq":
        prompt = f"Passage: {example['passage']}\nQuestion: {example['question']}\nAnswer:"
        return prompt, ["No", "Yes"], int(example["answer"])

    elif task == "piqa":
        prompt = f"Goal: {example['goal']}\nSolution:"
        return prompt, [example["sol1"], example["sol2"]], example["label"]

    elif task == "siqa":
        prompt = f"Context: {example['context']}\nQuestion: {example['question']}\nAnswer:"
        return prompt, [example["answerA"], example["answerB"], example["answerC"]], int(example["label"]) - 1

    elif task == "hellaswag":
        return example['ctx'], example["endings"], int(example["label"])

    elif task == "obqa":
        prompt = f"Question: {example['question_stem']}\nAnswer:"
        return prompt, example["choices"]["text"], ord(example["answerKey"]) - ord("A")

    elif task == "winogrande":
        prompt = example["sentence"].replace("_", "")
        return prompt, [example["option1"], example["option2"]], int(example["answer"]) - 1

    raise ValueError(f"Unknown task: {task}")
