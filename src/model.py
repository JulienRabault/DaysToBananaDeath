"""
Model architectures for banana ripeness classification.
Compatible with Hydra configuration system and SOLID architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchvision.models as models
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from typing import Optional, Dict, Any, List
import wandb
from pathlib import Path


class BaseBananaClassifier(L.LightningModule):
    """Base Lightning module for banana ripeness classification."""

    def __init__(
        self,
        num_classes: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = "cosine",  # "cosine", "step", "plateau", "none"
        class_weights: Optional[torch.Tensor] = None,
        log_wandb: bool = False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.log_wandb = log_wandb
        self.class_names = ['overripe', 'ripe', 'rotten', 'unripe']

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        self.val_confusion = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.test_confusion = ConfusionMatrix(task="multiclass", num_classes=num_classes)

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Model architecture (to be defined in subclasses)
        self.backbone = None
        self.classifier = None

    def forward(self, x):
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward method")

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.train_f1(preds, y)

        # Logging
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_confusion(preds, y)

        # Logging
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_confusion(preds, y)

        # Logging
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        """Log confusion matrix at end of validation epoch."""
        if self.log_wandb and self.logger and hasattr(self.logger, 'experiment'):
            cm = self.val_confusion.compute()
            self._log_confusion_matrix(cm, "val")
        self.val_confusion.reset()

    def on_test_epoch_end(self):
        """Log confusion matrix at end of test epoch."""
        if self.log_wandb and self.logger and hasattr(self.logger, 'experiment'):
            cm = self.test_confusion.compute()
            self._log_confusion_matrix(cm, "test")
        self.test_confusion.reset()

    def _log_confusion_matrix(self, cm, stage):
        """Log confusion matrix to wandb."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm.cpu().numpy(),
                annot=True,
                fmt='d',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=ax,
                cmap='Blues'
            )
            ax.set_title(f'{stage.capitalize()} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

            # Log to W&B if available
            if hasattr(self.logger.experiment, 'log'):
                self.logger.experiment.log({f"{stage}/confusion_matrix": wandb.Image(fig)})

            plt.close(fig)
        except ImportError:
            print("Matplotlib/Seaborn not available for confusion matrix plotting")
        except Exception as e:
            print(f"Error logging confusion matrix: {e}")

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        if self.scheduler == "none":
            return optimizer

        elif self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.learning_rate * 0.01
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch"
                }
            }

        elif self.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch"
                }
            }

        elif self.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=self.learning_rate * 0.01
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch"
                }
            }

        return optimizer

    def predict_step(self, batch, batch_idx):
        """Prediction step for inference."""
        x, _ = batch if isinstance(batch, tuple) else (batch, None)
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        return {"predictions": preds, "probabilities": probs, "logits": logits}


class ResNetClassifier(BaseBananaClassifier):
    """ResNet-based classifier for banana ripeness."""

    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout_rate

        # Get ResNet model
        if model_name == "resnet18":
            self.backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 512
        elif model_name == "resnet34":
            self.backbone = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 512
        elif model_name == "resnet50":
            self.backbone = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 2048
        elif model_name == "resnet101":
            self.backbone = models.resnet101(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")

        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class EfficientNetClassifier(BaseBananaClassifier):
    """EfficientNet-based classifier for banana ripeness."""

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout_rate

        # Get EfficientNet model
        if model_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 1280
        elif model_name == "efficientnet_b1":
            self.backbone = models.efficientnet_b1(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 1280
        elif model_name == "efficientnet_b2":
            self.backbone = models.efficientnet_b2(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 1408
        elif model_name == "efficientnet_b3":
            self.backbone = models.efficientnet_b3(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 1536
        else:
            raise ValueError(f"Unsupported EfficientNet model: {model_name}")

        # Remove the final classification layer
        self.backbone.classifier = nn.Identity()

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class VisionTransformerClassifier(BaseBananaClassifier):
    """Vision Transformer-based classifier for banana ripeness."""

    def __init__(
        self,
        model_name: str = "vit_b_16",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout_rate

        # Get Vision Transformer model
        if model_name == "vit_b_16":
            self.backbone = models.vit_b_16(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 768
        elif model_name == "vit_b_32":
            self.backbone = models.vit_b_32(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 768
        elif model_name == "vit_l_16":
            self.backbone = models.vit_l_16(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 1024
        else:
            raise ValueError(f"Unsupported ViT model: {model_name}")

        # Remove the final classification head
        self.backbone.heads = nn.Identity()

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class SimpleConvNet(BaseBananaClassifier):
    """Simple CNN for banana ripeness classification (lightweight option)."""

    def __init__(
        self,
        input_channels: int = 3,
        dropout_rate: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.input_channels = input_channels
        self.dropout_rate = dropout_rate

        # Simple CNN backbone
        self.backbone = nn.Sequential(
            # First block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        # Calculate the size after conv layers (assuming 224x224 input)
        # 224 -> 112 -> 56 -> 28 -> 14
        conv_output_size = 256 * 14 * 14

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class MobileNetClassifier(BaseBananaClassifier):
    """MobileNet-based classifier for banana ripeness (efficient mobile deployment)."""

    def __init__(
        self,
        model_name: str = "mobilenet_v3_small",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout_rate

        # Get MobileNet model
        if model_name == "mobilenet_v3_small":
            self.backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 576
        elif model_name == "mobilenet_v3_large":
            self.backbone = models.mobilenet_v3_large(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 960
        elif model_name == "mobilenet_v2":
            self.backbone = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
            backbone_dim = 1280
        else:
            raise ValueError(f"Unsupported MobileNet model: {model_name}")

        # Remove the final classification layer
        self.backbone.classifier = nn.Identity()

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, self.num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# Factory function to create models (legacy compatibility)
def create_model(
    model_type: str = "resnet50",
    num_classes: int = 4,
    pretrained: bool = True,
    **kwargs
) -> BaseBananaClassifier:
    """
    Factory function to create different model architectures.

    Args:
        model_type: Type of model
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        Configured model instance
    """
    model_type = model_type.lower()

    if model_type in ["resnet18", "resnet34", "resnet50", "resnet101"]:
        return ResNetClassifier(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    elif model_type in ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"]:
        return EfficientNetClassifier(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    elif model_type in ["vit_b_16", "vit_b_32", "vit_l_16"]:
        return VisionTransformerClassifier(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    elif model_type == "simple_cnn":
        return SimpleConvNet(
            num_classes=num_classes,
            **kwargs
        )
    elif model_type in ["mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"]:
        return MobileNetClassifier(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")

    # Test ResNet
    model = ResNetClassifier(model_name="resnet18", num_classes=4)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"ResNet18 output shape: {y.shape}")

    # Test EfficientNet
    model = EfficientNetClassifier(model_name="efficientnet_b0", num_classes=4)
    y = model(x)
    print(f"EfficientNet-B0 output shape: {y.shape}")

    # Test Simple CNN
    model = SimpleConvNet(num_classes=4)
    y = model(x)
    print(f"Simple CNN output shape: {y.shape}")

    print("All models working correctly!")
