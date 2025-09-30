"""
Custom callbacks for banana ripeness classification.
Each metric has its own callback class for better modularity.
"""

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from torchmetrics import Accuracy, F1Score, Recall, Precision, ConfusionMatrix
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AccuracyLoggerCallback(Callback):
    """Callback to log accuracy metrics during validation and test."""

    def __init__(
        self,
        log_on_step: bool = False,
        log_on_epoch: bool = True,
        prog_bar: bool = True,
        log_per_class: bool = False,
        num_classes: int = 4
    ):
        super().__init__()
        self.log_on_step = log_on_step
        self.log_on_epoch = log_on_epoch
        self.prog_bar = prog_bar
        self.log_per_class = log_per_class
        self.num_classes = num_classes

        # Class names for banana ripeness
        self.class_names = ['overripe', 'ripe', 'rotten', 'unripe']

        # Initialize metrics
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        if self.log_per_class:
            self.val_accuracy_per_class = Accuracy(task="multiclass", num_classes=num_classes, average=None)
            self.test_accuracy_per_class = Accuracy(task="multiclass", num_classes=num_classes, average=None)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Update accuracy after each validation batch."""
        if outputs is None:
            return

        x, y = batch
        with torch.no_grad():
            logits = pl_module(x)
            preds = torch.argmax(logits, dim=1)

        self.val_accuracy.update(preds, y)
        if self.log_per_class:
            self.val_accuracy_per_class.update(preds, y)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log accuracy at the end of validation epoch."""
        # Overall accuracy
        acc_value = self.val_accuracy.compute()
        pl_module.log("val/accuracy", acc_value,
                     on_step=self.log_on_step, on_epoch=self.log_on_epoch, prog_bar=self.prog_bar)
        self.val_accuracy.reset()

        # Per-class accuracy
        if self.log_per_class:
            per_class_acc = self.val_accuracy_per_class.compute()
            for i, class_name in enumerate(self.class_names):
                pl_module.log(f"val/accuracy_{class_name}", per_class_acc[i],
                             on_step=self.log_on_step, on_epoch=self.log_on_epoch)
            pl_module.log("val/accuracy_mean", per_class_acc.mean(),
                         on_step=self.log_on_step, on_epoch=self.log_on_epoch)
            self.val_accuracy_per_class.reset()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Update accuracy after each test batch."""
        if outputs is None:
            return

        x, y = batch
        with torch.no_grad():
            logits = pl_module(x)
            preds = torch.argmax(logits, dim=1)

        self.test_accuracy.update(preds, y)
        if self.log_per_class:
            self.test_accuracy_per_class.update(preds, y)

    def on_test_epoch_end(self, trainer, pl_module):
        """Log accuracy at the end of test epoch."""
        # Overall accuracy
        acc_value = self.test_accuracy.compute()
        pl_module.log("test/accuracy", acc_value, on_epoch=True)
        self.test_accuracy.reset()

        # Per-class accuracy
        if self.log_per_class:
            per_class_acc = self.test_accuracy_per_class.compute()
            for i, class_name in enumerate(self.class_names):
                pl_module.log(f"test/accuracy_{class_name}", per_class_acc[i], on_epoch=True)
            pl_module.log("test/accuracy_mean", per_class_acc.mean(), on_epoch=True)
            self.test_accuracy_per_class.reset()

    def setup(self, trainer, pl_module, stage):
        """Move metrics to correct device."""
        device = pl_module.device
        self.val_accuracy.to(device)
        self.test_accuracy.to(device)
        if self.log_per_class:
            self.val_accuracy_per_class.to(device)
            self.test_accuracy_per_class.to(device)


class F1LoggerCallback(Callback):
    """Callback to log F1-score metrics during validation and test."""

    def __init__(
        self,
        log_on_step: bool = False,
        log_on_epoch: bool = True,
        prog_bar: bool = True,
        log_per_class: bool = False,
        num_classes: int = 4
    ):
        super().__init__()
        self.log_on_step = log_on_step
        self.log_on_epoch = log_on_epoch
        self.prog_bar = prog_bar
        self.log_per_class = log_per_class
        self.num_classes = num_classes

        # Class names for banana ripeness
        self.class_names = ['overripe', 'ripe', 'rotten', 'unripe']

        # Initialize metrics
        self.val_f1_macro = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1_micro = F1Score(task="multiclass", num_classes=num_classes, average="micro")
        self.val_f1_weighted = F1Score(task="multiclass", num_classes=num_classes, average="weighted")

        self.test_f1_macro = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1_micro = F1Score(task="multiclass", num_classes=num_classes, average="micro")
        self.test_f1_weighted = F1Score(task="multiclass", num_classes=num_classes, average="weighted")

        if self.log_per_class:
            self.val_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)
            self.test_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Update F1 metrics after each validation batch."""
        if outputs is None:
            return

        x, y = batch
        with torch.no_grad():
            logits = pl_module(x)
            preds = torch.argmax(logits, dim=1)

        self.val_f1_macro.update(preds, y)
        self.val_f1_micro.update(preds, y)
        self.val_f1_weighted.update(preds, y)
        if self.log_per_class:
            self.val_f1_per_class.update(preds, y)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log F1 metrics at the end of validation epoch."""
        # Macro F1
        f1_macro = self.val_f1_macro.compute()
        pl_module.log("val/f1_macro", f1_macro,
                     on_step=self.log_on_step, on_epoch=self.log_on_epoch, prog_bar=self.prog_bar)
        self.val_f1_macro.reset()

        # Micro F1
        f1_micro = self.val_f1_micro.compute()
        pl_module.log("val/f1_micro", f1_micro,
                     on_step=self.log_on_step, on_epoch=self.log_on_epoch)
        self.val_f1_micro.reset()

        # Weighted F1
        f1_weighted = self.val_f1_weighted.compute()
        pl_module.log("val/f1_weighted", f1_weighted,
                     on_step=self.log_on_step, on_epoch=self.log_on_epoch)
        self.val_f1_weighted.reset()

        # Per-class F1
        if self.log_per_class:
            per_class_f1 = self.val_f1_per_class.compute()
            for i, class_name in enumerate(self.class_names):
                pl_module.log(f"val/f1_{class_name}", per_class_f1[i],
                             on_step=self.log_on_step, on_epoch=self.log_on_epoch)
            self.val_f1_per_class.reset()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Update F1 metrics after each test batch."""
        if outputs is None:
            return

        x, y = batch
        with torch.no_grad():
            logits = pl_module(x)
            preds = torch.argmax(logits, dim=1)

        self.test_f1_macro.update(preds, y)
        self.test_f1_micro.update(preds, y)
        self.test_f1_weighted.update(preds, y)
        if self.log_per_class:
            self.test_f1_per_class.update(preds, y)

    def on_test_epoch_end(self, trainer, pl_module):
        """Log F1 metrics at the end of test epoch."""
        # Macro F1
        f1_macro = self.test_f1_macro.compute()
        pl_module.log("test/f1_macro", f1_macro, on_epoch=True)
        self.test_f1_macro.reset()

        # Micro F1
        f1_micro = self.test_f1_micro.compute()
        pl_module.log("test/f1_micro", f1_micro, on_epoch=True)
        self.test_f1_micro.reset()

        # Weighted F1
        f1_weighted = self.test_f1_weighted.compute()
        pl_module.log("test/f1_weighted", f1_weighted, on_epoch=True)
        self.test_f1_weighted.reset()

        # Per-class F1
        if self.log_per_class:
            per_class_f1 = self.test_f1_per_class.compute()
            for i, class_name in enumerate(self.class_names):
                pl_module.log(f"test/f1_{class_name}", per_class_f1[i], on_epoch=True)
            self.test_f1_per_class.reset()

    def setup(self, trainer, pl_module, stage):
        """Move metrics to correct device."""
        device = pl_module.device
        self.val_f1_macro.to(device)
        self.val_f1_micro.to(device)
        self.val_f1_weighted.to(device)
        self.test_f1_macro.to(device)
        self.test_f1_micro.to(device)
        self.test_f1_weighted.to(device)
        if self.log_per_class:
            self.val_f1_per_class.to(device)
            self.test_f1_per_class.to(device)


class RecallLoggerCallback(Callback):
    """Callback to log recall metrics during validation and test."""

    def __init__(
        self,
        log_on_step: bool = False,
        log_on_epoch: bool = True,
        prog_bar: bool = True,
        log_per_class: bool = False,
        num_classes: int = 4
    ):
        super().__init__()
        self.log_on_step = log_on_step
        self.log_on_epoch = log_on_epoch
        self.prog_bar = prog_bar
        self.log_per_class = log_per_class
        self.num_classes = num_classes

        # Class names for banana ripeness
        self.class_names = ['overripe', 'ripe', 'rotten', 'unripe']

        # Initialize metrics
        self.val_recall_macro = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall_micro = Recall(task="multiclass", num_classes=num_classes, average="micro")
        self.val_recall_weighted = Recall(task="multiclass", num_classes=num_classes, average="weighted")

        self.test_recall_macro = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall_micro = Recall(task="multiclass", num_classes=num_classes, average="micro")
        self.test_recall_weighted = Recall(task="multiclass", num_classes=num_classes, average="weighted")

        if self.log_per_class:
            self.val_recall_per_class = Recall(task="multiclass", num_classes=num_classes, average=None)
            self.test_recall_per_class = Recall(task="multiclass", num_classes=num_classes, average=None)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Update recall metrics after each validation batch."""
        if outputs is None:
            return

        x, y = batch
        with torch.no_grad():
            logits = pl_module(x)
            preds = torch.argmax(logits, dim=1)

        self.val_recall_macro.update(preds, y)
        self.val_recall_micro.update(preds, y)
        self.val_recall_weighted.update(preds, y)
        if self.log_per_class:
            self.val_recall_per_class.update(preds, y)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log recall metrics at the end of validation epoch."""
        # Macro recall
        recall_macro = self.val_recall_macro.compute()
        pl_module.log("val/recall_macro", recall_macro,
                     on_step=self.log_on_step, on_epoch=self.log_on_epoch, prog_bar=self.prog_bar)
        self.val_recall_macro.reset()

        # Micro recall
        recall_micro = self.val_recall_micro.compute()
        pl_module.log("val/recall_micro", recall_micro,
                     on_step=self.log_on_step, on_epoch=self.log_on_epoch)
        self.val_recall_micro.reset()

        # Weighted recall
        recall_weighted = self.val_recall_weighted.compute()
        pl_module.log("val/recall_weighted", recall_weighted,
                     on_step=self.log_on_step, on_epoch=self.log_on_epoch)
        self.val_recall_weighted.reset()

        # Per-class recall
        if self.log_per_class:
            per_class_recall = self.val_recall_per_class.compute()
            for i, class_name in enumerate(self.class_names):
                pl_module.log(f"val/recall_{class_name}", per_class_recall[i],
                             on_step=self.log_on_step, on_epoch=self.log_on_epoch)
            self.val_recall_per_class.reset()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Update recall metrics after each test batch."""
        if outputs is None:
            return

        x, y = batch
        with torch.no_grad():
            logits = pl_module(x)
            preds = torch.argmax(logits, dim=1)

        self.test_recall_macro.update(preds, y)
        self.test_recall_micro.update(preds, y)
        self.test_recall_weighted.update(preds, y)
        if self.log_per_class:
            self.test_recall_per_class.update(preds, y)

    def on_test_epoch_end(self, trainer, pl_module):
        """Log recall metrics at the end of test epoch."""
        # Macro recall
        recall_macro = self.test_recall_macro.compute()
        pl_module.log("test/recall_macro", recall_macro, on_epoch=True)
        self.test_recall_macro.reset()

        # Micro recall
        recall_micro = self.test_recall_micro.compute()
        pl_module.log("test/recall_micro", recall_micro, on_epoch=True)
        self.test_recall_micro.reset()

        # Weighted recall
        recall_weighted = self.test_recall_weighted.compute()
        pl_module.log("test/recall_weighted", recall_weighted, on_epoch=True)
        self.test_recall_weighted.reset()

        # Per-class recall
        if self.log_per_class:
            per_class_recall = self.test_recall_per_class.compute()
            for i, class_name in enumerate(self.class_names):
                pl_module.log(f"test/recall_{class_name}", per_class_recall[i], on_epoch=True)
            self.test_recall_per_class.reset()

    def setup(self, trainer, pl_module, stage):
        """Move metrics to correct device."""
        device = pl_module.device
        self.val_recall_macro.to(device)
        self.val_recall_micro.to(device)
        self.val_recall_weighted.to(device)
        self.test_recall_macro.to(device)
        self.test_recall_micro.to(device)
        self.test_recall_weighted.to(device)
        if self.log_per_class:
            self.val_recall_per_class.to(device)
            self.test_recall_per_class.to(device)


class PrecisionLoggerCallback(Callback):
    """Callback to log precision metrics during validation and test."""

    def __init__(
        self,
        log_on_step: bool = False,
        log_on_epoch: bool = True,
        prog_bar: bool = False,
        log_per_class: bool = False,
        num_classes: int = 4
    ):
        super().__init__()
        self.log_on_step = log_on_step
        self.log_on_epoch = log_on_epoch
        self.prog_bar = prog_bar
        self.log_per_class = log_per_class
        self.num_classes = num_classes

        # Class names for banana ripeness
        self.class_names = ['overripe', 'ripe', 'rotten', 'unripe']

        # Initialize metrics
        self.val_precision_macro = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision_micro = Precision(task="multiclass", num_classes=num_classes, average="micro")
        self.val_precision_weighted = Precision(task="multiclass", num_classes=num_classes, average="weighted")

        self.test_precision_macro = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_precision_micro = Precision(task="multiclass", num_classes=num_classes, average="micro")
        self.test_precision_weighted = Precision(task="multiclass", num_classes=num_classes, average="weighted")

        if self.log_per_class:
            self.val_precision_per_class = Precision(task="multiclass", num_classes=num_classes, average=None)
            self.test_precision_per_class = Precision(task="multiclass", num_classes=num_classes, average=None)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Update precision metrics after each validation batch."""
        if outputs is None:
            return

        x, y = batch
        with torch.no_grad():
            logits = pl_module(x)
            preds = torch.argmax(logits, dim=1)

        self.val_precision_macro.update(preds, y)
        self.val_precision_micro.update(preds, y)
        self.val_precision_weighted.update(preds, y)
        if self.log_per_class:
            self.val_precision_per_class.update(preds, y)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log precision metrics at the end of validation epoch."""
        # Macro precision
        precision_macro = self.val_precision_macro.compute()
        pl_module.log("val/precision_macro", precision_macro,
                     on_step=self.log_on_step, on_epoch=self.log_on_epoch, prog_bar=self.prog_bar)
        self.val_precision_macro.reset()

        # Micro precision
        precision_micro = self.val_precision_micro.compute()
        pl_module.log("val/precision_micro", precision_micro,
                     on_step=self.log_on_step, on_epoch=self.log_on_epoch)
        self.val_precision_micro.reset()

        # Weighted precision
        precision_weighted = self.val_precision_weighted.compute()
        pl_module.log("val/precision_weighted", precision_weighted,
                     on_step=self.log_on_step, on_epoch=self.log_on_epoch)
        self.val_precision_weighted.reset()

        # Per-class precision
        if self.log_per_class:
            per_class_precision = self.val_precision_per_class.compute()
            for i, class_name in enumerate(self.class_names):
                pl_module.log(f"val/precision_{class_name}", per_class_precision[i],
                             on_step=self.log_on_step, on_epoch=self.log_on_epoch)
            self.val_precision_per_class.reset()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Update precision metrics after each test batch."""
        if outputs is None:
            return

        x, y = batch
        with torch.no_grad():
            logits = pl_module(x)
            preds = torch.argmax(logits, dim=1)

        self.test_precision_macro.update(preds, y)
        self.test_precision_micro.update(preds, y)
        self.test_precision_weighted.update(preds, y)
        if self.log_per_class:
            self.test_precision_per_class.update(preds, y)

    def on_test_epoch_end(self, trainer, pl_module):
        """Log precision metrics at the end of test epoch."""
        # Macro precision
        precision_macro = self.test_precision_macro.compute()
        pl_module.log("test/precision_macro", precision_macro, on_epoch=True)
        self.test_precision_macro.reset()

        # Micro precision
        precision_micro = self.test_precision_micro.compute()
        pl_module.log("test/precision_micro", precision_micro, on_epoch=True)
        self.test_precision_micro.reset()

        # Weighted precision
        precision_weighted = self.test_precision_weighted.compute()
        pl_module.log("test/precision_weighted", precision_weighted, on_epoch=True)
        self.test_precision_weighted.reset()

        # Per-class precision
        if self.log_per_class:
            per_class_precision = self.test_precision_per_class.compute()
            for i, class_name in enumerate(self.class_names):
                pl_module.log(f"test/precision_{class_name}", per_class_precision[i], on_epoch=True)
            self.test_precision_per_class.reset()

    def setup(self, trainer, pl_module, stage):
        """Move metrics to correct device."""
        device = pl_module.device
        self.val_precision_macro.to(device)
        self.val_precision_micro.to(device)
        self.val_precision_weighted.to(device)
        self.test_precision_macro.to(device)
        self.test_precision_micro.to(device)
        self.test_precision_weighted.to(device)
        if self.log_per_class:
            self.val_precision_per_class.to(device)
            self.test_precision_per_class.to(device)


# ...existing code...
