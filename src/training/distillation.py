"""Knowledge distillation training utilities."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from src.training.trainer import Trainer, TrainingConfig


@dataclass
class DistillationConfig(TrainingConfig):
    """Configuration for knowledge distillation training."""

    temperature: float = 2.0
    alpha: float = 0.5  # Weight for distillation loss vs task loss
    use_soft_targets: bool = True
    use_hidden_states: bool = False


class DistillationTrainer(Trainer):
    """Trainer for knowledge distillation from a larger teacher model."""

    def __init__(
        self,
        student_model: PreTrainedModel,
        teacher_model: PreTrainedModel,
        config: DistillationConfig,
        train_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader] = None,
        use_wandb: bool = True,
    ):
        super().__init__(
            model=student_model,
            config=config,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            use_wandb=use_wandb,
        )

        self.teacher_model = teacher_model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.distillation_config = config

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined distillation and task loss.

        Args:
            student_logits: Logits from student model.
            teacher_logits: Logits from teacher model.
            labels: Ground truth labels.

        Returns:
            Dictionary with loss components.
        """
        temperature = self.distillation_config.temperature
        alpha = self.distillation_config.alpha

        # Soft target loss (KL divergence)
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        soft_loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction="batchmean",
        ) * (temperature ** 2)

        # Hard target loss (cross entropy)
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        hard_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Combined loss
        total_loss = alpha * soft_loss + (1 - alpha) * hard_loss

        return {
            "total_loss": total_loss,
            "soft_loss": soft_loss,
            "hard_loss": hard_loss,
        }

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute a single training step with distillation.

        Args:
            batch: Input batch.

        Returns:
            Dictionary with losses.
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**batch)
            teacher_logits = teacher_outputs.logits

        student_outputs = self.model(**batch)
        student_logits = student_outputs.logits

        losses = self.compute_distillation_loss(
            student_logits,
            teacher_logits,
            batch["labels"],
        )

        return losses
