"""
Custom callbacks for banana ripeness classification.
Includes metrics logging and other training utilities.
"""

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from torchmetrics import Accuracy, F1Score, Recall, Precision, ConfusionMatrix
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsLoggerCallback(Callback):
    """
    Custom callback to log detailed metrics during validation.
    Tracks accuracy, F1-score, recall, and precision.
    """

    def __init__(
        self,
        log_accuracy: bool = True,
        log_f1: bool = True,
        log_recall: bool = True,
        log_precision: bool = True,
        log_confusion_matrix: bool = False,
        log_on_step: bool = False,
        log_on_epoch: bool = True,
        prog_bar: bool = True,
        log_per_class: bool = False,
        num_classes: int = 4
    ):
        """
        Initialize the metrics logger callback.

        Args:
            log_accuracy: Whether to log accuracy
            log_f1: Whether to log F1-score
            log_recall: Whether to log recall
            log_precision: Whether to log precision
            log_confusion_matrix: Whether to log confusion matrix
            log_on_step: Log metrics on each step
            log_on_epoch: Log metrics on each epoch
            prog_bar: Show metrics in progress bar
            log_per_class: Log per-class metrics
            num_classes: Number of classes
        """
        super().__init__()

        self.log_accuracy = log_accuracy
        self.log_f1 = log_f1
        self.log_recall = log_recall
        self.log_precision = log_precision
        self.log_confusion_matrix = log_confusion_matrix
        self.log_on_step = log_on_step
        self.log_on_epoch = log_on_epoch
        self.prog_bar = prog_bar
        self.log_per_class = log_per_class
        self.num_classes = num_classes

        # Class names for banana ripeness
        self.class_names = ['overripe', 'ripe', 'rotten', 'unripe']

        # Initialize metrics
        self.val_metrics = {}
        self.test_metrics = {}

        # Validation metrics
        if self.log_accuracy:
            self.val_metrics['accuracy'] = Accuracy(task="multiclass", num_classes=num_classes)
            if self.log_per_class:
                self.val_metrics['accuracy_per_class'] = Accuracy(
                    task="multiclass", num_classes=num_classes, average=None
                )

        if self.log_f1:
            self.val_metrics['f1_macro'] = F1Score(task="multiclass", num_classes=num_classes, average="macro")
            self.val_metrics['f1_micro'] = F1Score(task="multiclass", num_classes=num_classes, average="micro")
            self.val_metrics['f1_weighted'] = F1Score(task="multiclass", num_classes=num_classes, average="weighted")
            if self.log_per_class:
                self.val_metrics['f1_per_class'] = F1Score(
                    task="multiclass", num_classes=num_classes, average=None
                )

        if self.log_recall:
            self.val_metrics['recall_macro'] = Recall(task="multiclass", num_classes=num_classes, average="macro")
            self.val_metrics['recall_micro'] = Recall(task="multiclass", num_classes=num_classes, average="micro")
            self.val_metrics['recall_weighted'] = Recall(task="multiclass", num_classes=num_classes, average="weighted")
            if self.log_per_class:
                self.val_metrics['recall_per_class'] = Recall(
                    task="multiclass", num_classes=num_classes, average=None
                )

        if self.log_precision:
            self.val_metrics['precision_macro'] = Precision(task="multiclass", num_classes=num_classes, average="macro")
            self.val_metrics['precision_micro'] = Precision(task="multiclass", num_classes=num_classes, average="micro")
            self.val_metrics['precision_weighted'] = Precision(task="multiclass", num_classes=num_classes, average="weighted")
            if self.log_per_class:
                self.val_metrics['precision_per_class'] = Precision(
                    task="multiclass", num_classes=num_classes, average=None
                )

        if self.log_confusion_matrix:
            self.val_metrics['confusion_matrix'] = ConfusionMatrix(task="multiclass", num_classes=num_classes)

        # Test metrics (copy of validation metrics)
        self.test_metrics = {f"test_{k}": v.clone() for k, v in self.val_metrics.items()}

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0
    ):
        """Update metrics after each validation batch."""
        if outputs is None:
            return

        # Get batch data
        x, y = batch

        # Run forward pass to get logits (since outputs only contains loss)
        with torch.no_grad():
            logits = pl_module(x)
            preds = torch.argmax(logits, dim=1)

        # Update all validation metrics
        for metric in self.val_metrics.values():
            metric = metric.to(preds.device)
            metric.update(preds, y)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Log metrics at the end of validation epoch."""

        # Compute and log all metrics
        for metric_name, metric in self.val_metrics.items():
            try:
                value = metric.compute()

                if metric_name == 'confusion_matrix':
                    # Special handling for confusion matrix
                    if self.log_confusion_matrix:
                        # Log as a table or matrix format if logger supports it
                        pl_module.log(f"val/{metric_name}", value,
                                    on_step=self.log_on_step, on_epoch=self.log_on_epoch)

                elif 'per_class' in metric_name:
                    # Log per-class metrics with class names
                    if self.log_per_class and value.numel() == len(self.class_names):
                        for i, class_name in enumerate(self.class_names):
                            class_metric_name = f"val/{metric_name}_{class_name}"
                            pl_module.log(class_metric_name, value[i],
                                        on_step=self.log_on_step, on_epoch=self.log_on_epoch)

                        # Also log the mean
                        pl_module.log(f"val/{metric_name}_mean", value.mean(),
                                    on_step=self.log_on_step, on_epoch=self.log_on_epoch,
                                    prog_bar=self.prog_bar)

                else:
                    # Regular scalar metrics
                    pl_module.log(f"val/{metric_name}", value,
                                on_step=self.log_on_step, on_epoch=self.log_on_epoch,
                                prog_bar=self.prog_bar and metric_name in ['accuracy', 'f1_macro', 'recall_macro'])

                # Reset metric for next epoch
                metric.reset()

            except Exception as e:
                logger.warning(f"Failed to compute metric {metric_name}: {e}")

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0
    ):
        """Update metrics after each test batch."""
        if outputs is None:
            return

        # Get batch data
        x, y = batch

        # Run forward pass to get logits (since outputs only contains loss)
        with torch.no_grad():
            logits = pl_module(x)
            preds = torch.argmax(logits, dim=1)

        # Update all test metrics
        for metric in self.test_metrics.values():
            metric = metric.to(preds.device)
            metric.update(preds, y)

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Log metrics at the end of test epoch."""

        # Compute and log all test metrics
        for metric_name, metric in self.test_metrics.items():
            try:
                value = metric.compute()

                if 'confusion_matrix' in metric_name:
                    # Special handling for confusion matrix
                    if self.log_confusion_matrix:
                        pl_module.log(metric_name, value, on_epoch=True)

                elif 'per_class' in metric_name:
                    # Log per-class metrics with class names
                    if self.log_per_class and value.numel() == len(self.class_names):
                        for i, class_name in enumerate(self.class_names):
                            class_metric_name = f"{metric_name}_{class_name}"
                            pl_module.log(class_metric_name, value[i], on_epoch=True)

                        # Also log the mean
                        pl_module.log(f"{metric_name}_mean", value.mean(), on_epoch=True)

                else:
                    # Regular scalar metrics
                    pl_module.log(metric_name, value, on_epoch=True)

                # Reset metric for next epoch
                metric.reset()

            except Exception as e:
                logger.warning(f"Failed to compute test metric {metric_name}: {e}")

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str):
        """Setup callback - move metrics to correct device."""
        device = pl_module.device

        # Move validation metrics to device
        for metric in self.val_metrics.values():
            metric.to(device)

        # Move test metrics to device
        for metric in self.test_metrics.values():
            metric.to(device)


class ModelSummaryCallback(Callback):
    """
    Callback to log model summary and parameter count.
    """

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Log model summary when training starts."""

        # Count parameters
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        # Log parameter counts
        pl_module.log("model/total_parameters", float(total_params), on_epoch=False)
        pl_module.log("model/trainable_parameters", float(trainable_params), on_epoch=False)
        pl_module.log("model/non_trainable_parameters", float(non_trainable_params), on_epoch=False)

        # Print summary
        logger.info(f"Model Summary:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Non-trainable parameters: {non_trainable_params:,}")
        logger.info(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")


class LearningRateLoggerCallback(Callback):
    """
    Enhanced learning rate logger with additional optimizer info.
    """

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx):
        """Log learning rate after each training batch."""

        # Get current learning rate from optimizer
        if trainer.optimizers:
            optimizer = trainer.optimizers[0]
            for param_group_idx, param_group in enumerate(optimizer.param_groups):
                lr = param_group['lr']
                pl_module.log(f"train/lr_group_{param_group_idx}", lr,
                            on_step=True, on_epoch=False, prog_bar=False)

                # Log main learning rate to progress bar
                if param_group_idx == 0:
                    pl_module.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=True)
