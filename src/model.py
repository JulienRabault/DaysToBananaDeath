"""
Model architectures for banana ripeness classification.
Compatible with Hydra configuration system and SOLID architecture.
"""

import torch
import torch.nn as nn
import lightning as L
import torchvision.models as models
import torchmetrics
from typing import Optional
import ultralytics
from ultralytics import YOLO
import torchvision.transforms as transforms


class BaseBananaClassifier(L.LightningModule):
    """Base Lightning module for banana ripeness classification."""

    def __init__(
        self,
        num_classes: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = "cosine",  # "cosine", "step", "plateau", "none"
        class_weights: Optional[torch.Tensor] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.class_names = ['overripe', 'ripe', 'rotten', 'unripe']

        # Metrics collections
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes),
                "f1": torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average="macro"),
                "precision": torchmetrics.classification.Precision(task="multiclass", num_classes=num_classes, average="macro"),
                "recall": torchmetrics.classification.Recall(task="multiclass", num_classes=num_classes, average="macro"),
            },
            prefix="train/",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

        # Loss function
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

        # Update and log metrics
        preds = torch.argmax(logits, dim=1)
        batch_metrics = self.train_metrics(preds, y)

        # Log loss and metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(batch_metrics, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        """Reset training metrics at end of epoch."""
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Update metrics
        preds = torch.argmax(logits, dim=1)
        self.val_metrics.update(preds, y)

        # Log only loss here, metrics logged in on_validation_epoch_end
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        """Compute and log validation metrics, then reset."""
        val_metrics = self.val_metrics.compute()
        self.log_dict(val_metrics, on_epoch=True, prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Update metrics
        preds = torch.argmax(logits, dim=1)
        self.test_metrics.update(preds, y)

        # Log only loss here, metrics logged in on_test_epoch_end
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        """Compute and log test metrics, then reset."""
        test_metrics = self.test_metrics.compute()
        self.log_dict(test_metrics, on_epoch=True)
        self.test_metrics.reset()

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
    """ResNet50-based classifier for banana ripeness."""

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

        # Only support resnet50
        if model_name != "resnet50":
            raise ValueError(f"Unsupported ResNet model: {model_name}. Only 'resnet50' is supported.")
        self.backbone = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        backbone_dim = 2048

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


class VisionTransformerClassifier(BaseBananaClassifier):
    """Vision Transformer (ViT-B/16)-based classifier for banana ripeness."""

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

        # Only support vit_b_16
        if model_name != "vit_b_16":
            raise ValueError(f"Unsupported ViT model: {model_name}. Only 'vit_b_16' is supported.")
        self.backbone = models.vit_b_16(weights="IMAGENET1K_V1" if pretrained else None)
        backbone_dim = 768

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


class YOLOClassifier(BaseBananaClassifier):
    """YOLO-based classifier for banana ripeness detection and classification."""

    def __init__(
        self,
        model_name: str = "yolov8n",
        pretrained: bool = True,
        confidence_threshold: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model_name = model_name
        self.pretrained = pretrained
        self.confidence_threshold = confidence_threshold

        # Initialize YOLO model for detection
        if pretrained:
            self.yolo_detector = YOLO(f'{model_name}.pt')
        else:
            self.yolo_detector = YOLO(f'{model_name}.yaml')

        # Custom classification head for ripeness classification
        # YOLO feature dimension depends on model size
        feature_dims = {
            'yolov8n': 512,
            'yolov8s': 512,
            'yolov8m': 768,
            'yolov8l': 1024,
            'yolov8x': 1280
        }
        backbone_dim = feature_dims.get(model_name, 512)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )

        # Transform to convert PIL images to tensor for YOLO
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
        ])

    def extract_features_from_detection(self, image, detection_box):
        """Extract features from the detected bounding box region."""
        if detection_box is None:
            # If no detection, use the whole image
            crop = image
        else:
            # Crop the image using the detection box
            x1, y1, x2, y2 = detection_box
            crop = image[:, :, int(y1):int(y2), int(x1):int(x2)]

        # Resize crop to standard size for feature extraction
        crop_resized = torch.nn.functional.interpolate(crop, size=(224, 224), mode='bilinear', align_corners=False)

        # Use YOLO backbone to extract features
        with torch.no_grad():
            # Get intermediate features from YOLO model
            features = self.yolo_detector.model.model[:9](crop_resized)  # Use first 9 layers for feature extraction

        return features

    def forward(self, x):
        """Forward pass using YOLO detection + classification."""
        batch_size = x.shape[0]
        batch_logits = []

        for i in range(batch_size):
            single_image = x[i:i+1]

            # Convert tensor to PIL for YOLO detection
            image_pil = transforms.ToPILImage()(single_image.squeeze(0))

            # Run YOLO detection
            with torch.no_grad():
                results = self.yolo_detector(image_pil, verbose=False)

            # Get the first detection above confidence threshold
            detection_box = None
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()

                # Find first detection above threshold
                valid_detections = confidences >= self.confidence_threshold
                if valid_detections.any():
                    first_valid_idx = valid_detections.argmax()
                    detection_box = boxes.xyxy[first_valid_idx].cpu().numpy()

            # Extract features from detection region
            features = self.extract_features_from_detection(single_image, detection_box)

            # Classify the detected region
            logits = self.classifier(features)
            batch_logits.append(logits)

        # Stack all logits from the batch
        return torch.cat(batch_logits, dim=0)

    def predict_step(self, batch, batch_idx):
        """Prediction step for inference with detection information."""
        x, _ = batch if isinstance(batch, tuple) else (batch, None)

        # Get detections and classifications
        detections_info = []
        batch_logits = []

        for i in range(x.shape[0]):
            single_image = x[i:i+1]

            # Convert tensor to PIL for YOLO detection
            image_pil = transforms.ToPILImage()(single_image.squeeze(0))

            # Run YOLO detection
            with torch.no_grad():
                results = self.yolo_detector(image_pil, verbose=False)

            # Get the first detection above confidence threshold
            detection_info = {"has_detection": False, "box": None, "confidence": 0.0}
            detection_box = None

            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()

                # Find first detection above threshold
                valid_detections = confidences >= self.confidence_threshold
                if valid_detections.any():
                    first_valid_idx = valid_detections.argmax()
                    detection_box = boxes.xyxy[first_valid_idx].cpu().numpy()
                    detection_info = {
                        "has_detection": True,
                        "box": detection_box.tolist(),
                        "confidence": float(confidences[first_valid_idx])
                    }

            detections_info.append(detection_info)

            # Extract features and classify
            features = self.extract_features_from_detection(single_image, detection_box)
            logits = self.classifier(features)
            batch_logits.append(logits)

        # Stack all logits
        logits = torch.cat(batch_logits, dim=0)
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        return {
            "predictions": preds,
            "probabilities": probs,
            "logits": logits,
            "detections": detections_info
        }
