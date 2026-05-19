import os
import torch
import logging
import wandb
from typing import Dict, Optional, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Unified logger for WandB monitoring and local checkpoint management.

    Tracks training metrics, loss curves, and saves best model weights
    based on validation LSE-C metric.
    """

    def __init__(
        self,
        config: Dict,
        project_name: str = 'eventspeech',
        run_name: Optional[str] = None,
        checkpoint_dir: str = 'checkpoints',
        use_wandb: bool = True,
    ):
        self.config = config
        self.project_name = project_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.use_wandb = use_wandb

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize WandB
        if use_wandb:
            run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=project_name,
                name=run_name,
                config=config,
                reinit=True,
            )

        # Best model tracking
        self.best_metric = float('-inf')
        self.best_epoch = -1

        # Setup file logging
        self._setup_file_logging()

    def _setup_file_logging(self):
        """Setup file-based logging."""
        log_file = self.checkpoint_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = '',
    ):
        """Log metrics to WandB and local log.

        Args:
            metrics: Dictionary of metric name to value
            step: Current training step
            prefix: Optional prefix for metric names
        """
        # Add prefix to metric names
        logged_metrics = {}
        for key, value in metrics.items():
            metric_name = f"{prefix}/{key}" if prefix else key
            logged_metrics[metric_name] = value

        # Log to WandB
        if self.use_wandb:
            wandb.log(logged_metrics, step=step)

        # Log to file
        metric_str = ', '.join([f"{k}: {v:.6f}" for k, v in logged_metrics.items()])
        logger.info(f"Step {step} - {metric_str}")

    def log_loss(
        self,
        losses: Dict[str, torch.Tensor],
        step: int,
        prefix: str = 'train',
    ):
        """Log loss values.

        Args:
            losses: Dictionary of loss name to tensor value
            step: Current training step
            prefix: Prefix for loss names
        """
        loss_dict = {}
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                loss_dict[f"{prefix}/loss/{key}"] = value.item()
            else:
                loss_dict[f"{prefix}/loss/{key}"] = float(value)

        if self.use_wandb:
            wandb.log(loss_dict, step=step)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ):
        """Save model checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            epoch: Current epoch
            step: Current step
            metrics: Current metrics
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch}")

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)

    def update_best_model(
        self,
        metric_value: float,
        epoch: int,
        metric_name: str = 'lse_c',
    ) -> bool:
        """Check if current model is the best.

        Args:
            metric_value: Current metric value
            epoch: Current epoch
            metric_name: Name of the metric to track

        Returns:
            True if this is the best model so far
        """
        if metric_value > self.best_metric:
            self.best_metric = metric_value
            self.best_epoch = epoch
            logger.info(
                f"New best {metric_name}: {metric_value:.6f} at epoch {epoch}"
            )
            return True
        return False

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict:
        """Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights
            optimizer: Optimizer to load state (optional)
            scheduler: Scheduler to load state (optional)

        Returns:
            Dictionary with checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

        return {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
            'metrics': checkpoint.get('metrics', {}),
        }

    def finish(self):
        """Finish logging session."""
        if self.use_wandb:
            wandb.finish()
        logger.info("Logging session finished")
