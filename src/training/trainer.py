"""Main trainer implementation."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, get_scheduler
from tqdm import tqdm
import wandb

from src.training.optimizer import create_optimizer
from src.training.scheduler import create_scheduler


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Schedule
    warmup_steps: int = 1000
    max_steps: int = 100000
    scheduler_type: str = "cosine"
    min_lr_ratio: float = 0.1

    # Batch
    gradient_accumulation_steps: int = 4

    # Logging
    logging_steps: int = 100
    validation_steps: int = 1000
    save_steps: int = 5000

    # Mixed precision
    fp16: bool = False
    bf16: bool = True

    # Misc
    seed: int = 42
    output_dir: str = "experiments"


class Trainer:
    """Trainer for language model training with scaling law tracking."""

    def __init__(
        self,
        model: PreTrainedModel,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader] = None,
        use_wandb: bool = True,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.use_wandb = use_wandb

        self.device = next(model.parameters()).device
        self.global_step = 0
        self.total_tokens = 0

        self._setup_training()

    def _setup_training(self):
        """Initialize optimizer, scheduler, and mixed precision."""
        self.optimizer = create_optimizer(
            self.model,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=self.config.scheduler_type,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps,
            min_lr_ratio=self.config.min_lr_ratio,
        )

        self.scaler = None
        if self.config.fp16:
            self.scaler = torch.amp.GradScaler("cuda")

    def train(self) -> Dict[str, Any]:
        """Run training loop."""
        self.model.train()

        progress_bar = tqdm(total=self.config.max_steps, desc="Training")

        accumulated_loss = 0.0
        num_accumulated = 0

        data_iter = iter(self.train_dataloader)

        while self.global_step < self.config.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.amp.autocast(
                "cuda",
                enabled=self.config.fp16 or self.config.bf16,
                dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            ):
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item()
            num_accumulated += 1

            if num_accumulated >= self.config.gradient_accumulation_steps:
                if self.config.max_grad_norm > 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                self.total_tokens += (
                    batch["input_ids"].numel() * self.config.gradient_accumulation_steps
                )

                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = accumulated_loss * self.config.gradient_accumulation_steps
                    self._log_metrics({
                        "train/loss": avg_loss,
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/global_step": self.global_step,
                        "train/total_tokens": self.total_tokens,
                    })

                accumulated_loss = 0.0
                num_accumulated = 0

                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{loss.item() * self.config.gradient_accumulation_steps:.4f}")

                if self.global_step % self.config.validation_steps == 0 and self.validation_dataloader:
                    self.validate()

        progress_bar.close()

        return {"final_step": self.global_step, "total_tokens": self.total_tokens}

    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(self.validation_dataloader, desc="Validating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.amp.autocast(
                    "cuda",
                    enabled=self.config.fp16 or self.config.bf16,
                    dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                ):
                    outputs = self.model(**batch)

                total_loss += outputs.loss.item() * batch["input_ids"].numel()
                total_tokens += batch["input_ids"].numel()

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        metrics = {"validation/loss": avg_loss, "validation/perplexity": perplexity}
        self._log_metrics(metrics)

        self.model.train()

        return metrics

    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to wandb if enabled."""
        if self.use_wandb and wandb.run is not None:
            wandb.log(metrics, step=self.global_step)
