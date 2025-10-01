"""
Trainer helper utilities for banana ripeness classification.
Implements SOLID principles for clean and maintainable training pipeline.
"""

import os
import sys
import torch
import wandb
import lightning as L
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from omegaconf import DictConfig
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import Callback
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from datetime import datetime
from uniqpath import unique_path


class ConfigValidator:
    """Validates and sanitizes configuration objects."""

    @staticmethod
    def validate_config(cfg: DictConfig) -> None:
        """Validate configuration for required fields and correct types."""
        required_sections = ['model', 'data', 'trainer', 'logger']

        for section in required_sections:
            if section not in cfg:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate model config
        if 'num_classes' not in cfg.model:
            cfg.model.num_classes = 4

        # Validate data config
        if 'batch_size' not in cfg.data:
            cfg.data.batch_size = 32

        # Validate trainer config
        if 'max_epochs' not in cfg.trainer:
            cfg.trainer.max_epochs = 50


class ExperimentManager:
    """Manages experiment metadata and versioning following ML best practices."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.experiment_name = self._generate_experiment_name()
        self.run_name = self._generate_run_name()
        self.version = self._generate_version()
        self.output_dir = self._setup_output_directory()

    def _generate_experiment_name(self) -> str:
        """Generate experiment name following ML naming conventions."""
        # Use explicit experiment name if provided
        if hasattr(self.cfg, 'experiment') and self.cfg.experiment.get('name'):
            return self.cfg.experiment.name

        # Default: dataset-model pattern (ML standard)
        model_name = self.cfg.model.get('model_name', 'unknown')
        dataset_name = self.cfg.data.get('dataset_name', 'banana-ripeness')
        return f"{dataset_name}-{model_name}"

    def _generate_run_name(self) -> str:
        """Generate descriptive run name with key hyperparameters using uniqpath for uniqueness."""
        components = []

        # Model architecture
        model_name = self.cfg.model.get('model_name', 'unknown')
        components.append(model_name)

        # Key hyperparameters (ML naming convention)
        lr = self.cfg.model.get('learning_rate', 1e-3)
        batch_size = self.cfg.data.get('batch_size', 32)
        epochs = self.cfg.trainer.get('max_epochs', 50)

        # Format: model_lr1e-3_bs32_ep50
        components.append(f"lr{lr:.0e}")
        components.append(f"bs{batch_size}")
        components.append(f"ep{epochs}")

        # Add experiment tag if provided
        if hasattr(self.cfg, 'experiment') and self.cfg.experiment.get('tag'):
            components.append(self.cfg.experiment.tag)

        # Base run name without uniqueness suffix
        base_run_name = "_".join(components)

        # Use uniqpath to ensure uniqueness with timestamp format
        # This will add _{timestamp} if needed to avoid conflicts
        return unique_path(
            base_run_name,
            suffix_format="_{timestamp}",
            if_exists_only=False,  # Always add suffix for better organization
            return_str=True
        ) if hasattr(unique_path(base_run_name, suffix_format="_{timestamp}", if_exists_only=False, return_str=False), 'name') else str(unique_path(base_run_name, suffix_format="_{timestamp}", if_exists_only=False, return_str=True))

    def _generate_version(self) -> str:
        """Generate semantic version following ML versioning best practices."""
        if hasattr(self.cfg, 'experiment') and self.cfg.experiment.get('version'):
            return self.cfg.experiment.version

        # Auto-generate version based on date (semantic versioning for ML)
        return datetime.now().strftime("v%Y.%m.%d")

    def _setup_output_directory(self) -> Path:
        """Setup hierarchical output directory structure using uniqpath."""
        # Structure: outputs/experiment_name/version/run_name
        base_path = Path(f"outputs/{self.experiment_name}/{self.version}/{self.run_name}")

        # Use uniqpath to ensure the directory path is unique
        unique_output_dir = unique_path(
            base_path,
            suffix_format="_{rand:4}",  # Short random suffix if needed
            if_exists_only=True,        # Only add suffix if directory exists
            return_str=False
        )

        unique_output_dir.mkdir(parents=True, exist_ok=True)

        # Create standard MLOps subdirectories
        (unique_output_dir / "checkpoints").mkdir(exist_ok=True)
        (unique_output_dir / "logs").mkdir(exist_ok=True)
        (unique_output_dir / "artifacts").mkdir(exist_ok=True)
        (unique_output_dir / "metrics").mkdir(exist_ok=True)
        (unique_output_dir / "plots").mkdir(exist_ok=True)

        return unique_output_dir

    def get_run_tags(self) -> List[str]:
        """Generate standardized tags for the run."""
        tags = []

        # Model architecture tag
        model_name = self.cfg.model.get('model_name', 'unknown')
        tags.append(f"model:{model_name}")

        # Task type
        tags.append("task:classification")
        tags.append("domain:computer-vision")

        # Dataset
        dataset_name = self.cfg.data.get('dataset_name', 'banana-ripeness')
        tags.append(f"dataset:{dataset_name}")

        # Training config
        scheduler = self.cfg.model.get('scheduler', 'cosine')
        tags.append(f"scheduler:{scheduler}")

        # Custom tags from config
        if hasattr(self.cfg, 'experiment') and self.cfg.experiment.get('tags'):
            tags.extend(self.cfg.experiment.tags)

        return tags

    def get_run_metadata(self) -> Dict[str, any]:
        """Generate comprehensive metadata for the run."""
        return {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "version": self.version,
            "model_architecture": self.cfg.model.get('model_name'),
            "dataset": self.cfg.data.get('dataset_name', 'banana-ripeness'),
            "num_classes": self.cfg.model.get('num_classes', 4),
            "hyperparameters": {
                "learning_rate": self.cfg.model.get('learning_rate'),
                "batch_size": self.cfg.data.get('batch_size'),
                "max_epochs": self.cfg.trainer.get('max_epochs'),
                "optimizer": "AdamW",
                "scheduler": self.cfg.model.get('scheduler'),
                "weight_decay": self.cfg.model.get('weight_decay'),
            },
            "framework": "pytorch-lightning",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "created_at": datetime.now().isoformat(),
        }

    # Legacy methods for backward compatibility
    def get_checkpoint_dir(self) -> str:
        """Get checkpoint directory path."""
        return str(self.output_dir / "checkpoints")

    def get_logs_dir(self) -> str:
        """Get logs directory path."""
        return str(self.output_dir / "logs")

    def get_artifacts_dir(self) -> str:
        """Get artifacts directory path."""
        return str(self.output_dir / "artifacts")


class ComponentFactory:
    """Factory for creating training components with dependency injection."""

    @staticmethod
    def create_datamodule(cfg: DictConfig) -> L.LightningDataModule:
        """Create data module from configuration."""
        return instantiate(cfg.data)

    @staticmethod
    def create_model(cfg: DictConfig, class_weights: Optional[torch.Tensor] = None) -> L.LightningModule:
        """Create model from configuration."""
        model_cfg = cfg.model.copy()
        return instantiate(model_cfg, class_weights=class_weights)

    @staticmethod
    def create_logger(cfg: DictConfig, experiment_manager: ExperimentManager) -> Logger:
        """Create logger from configuration with proper ML naming."""
        logger_cfg = cfg.logger.copy()

        # Use the run_name (descriptive) for WandB, not experiment_name
        logger_cfg.name = experiment_manager.run_name
        logger_cfg.save_dir = experiment_manager.get_logs_dir()

        # Set standardized tags and metadata
        logger_cfg.tags = experiment_manager.get_run_tags()

        # Set experiment group for organization
        if hasattr(logger_cfg, 'group'):
            logger_cfg.group = experiment_manager.experiment_name

        # Add version info
        if hasattr(logger_cfg, 'version'):
            logger_cfg.version = experiment_manager.version

        # Set notes from metadata
        metadata = experiment_manager.get_run_metadata()
        if hasattr(logger_cfg, 'notes'):
            logger_cfg.notes = f"Model: {metadata['model_architecture']}, LR: {metadata['hyperparameters']['learning_rate']}, BS: {metadata['hyperparameters']['batch_size']}"

        return instantiate(logger_cfg)

    @staticmethod
    def create_callbacks(cfg: DictConfig, experiment_manager: ExperimentManager) -> List[Callback]:
        """Create callbacks from configuration."""
        callbacks: List[Callback] = []

        if 'callbacks' in cfg:
            for callback_name, callback_cfg in cfg.callbacks.items():
                if callback_name.startswith('_'):
                    continue

                # Special handling for ModelCheckpoint
                if 'model_checkpoint' in callback_name and hasattr(callback_cfg, 'dirpath'):
                    if callback_cfg.dirpath is None:
                        callback_cfg.dirpath = experiment_manager.get_checkpoint_dir()

                callback = instantiate(callback_cfg)
                callbacks.append(callback)

        return callbacks

    @staticmethod
    def create_trainer(cfg: DictConfig, logger: Logger, callbacks: List[Callback]) -> L.Trainer:
        """Create trainer from configuration."""
        trainer_cfg = cfg.trainer.copy()

        # No need to sync with Hydra anymore - ExperimentManager handles directory structure
        # The trainer will use the logger's save_dir which is already set by ExperimentManager

        return instantiate(trainer_cfg, logger=logger, callbacks=callbacks)


class ModelArtifactManager:
    """Manages model artifacts and W&B integration with ML naming conventions."""

    def __init__(self, experiment_manager: ExperimentManager, cfg: DictConfig):
        self.experiment_manager = experiment_manager
        self.cfg = cfg

    def save_model_artifacts(self, model: L.LightningModule, trainer: L.Trainer) -> None:
        """Save model artifacts to W&B and local storage."""
        if not self._is_wandb_enabled():
            return

        artifacts_dir = Path(self.experiment_manager.get_artifacts_dir())

        # Get best checkpoint path
        best_checkpoint_path = self._get_best_checkpoint(trainer)
        if not best_checkpoint_path:
            print("Warning: No best checkpoint found")
            return

        # Load best model
        best_model = type(model).load_from_checkpoint(best_checkpoint_path)

        # Save different model formats
        self._save_state_dict(best_model, artifacts_dir)
        self._save_full_model(best_model, artifacts_dir)
        self._save_onnx_model(best_model, artifacts_dir)

        # Create W&B artifact with proper naming
        self._create_wandb_artifact(artifacts_dir, trainer, best_checkpoint_path)

    def _is_wandb_enabled(self) -> bool:
        """Check if W&B logging is enabled."""
        return (hasattr(self.cfg.logger, '_target_') and
                'WandbLogger' in self.cfg.logger._target_)

    def _get_best_checkpoint(self, trainer: L.Trainer) -> Optional[str]:
        """Get path to best checkpoint (robust across Lightning versions)."""
        # Prefer scanning callbacks for ModelCheckpoint
        best_path: Optional[str] = None
        for cb in getattr(trainer, 'callbacks', []) or []:
            if isinstance(cb, ModelCheckpoint):
                if getattr(cb, 'best_model_path', None):
                    best_path = cb.best_model_path
                    break
        # Fallback to trainer.checkpoint_callback if available
        if not best_path and hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
            best_path = getattr(trainer.checkpoint_callback, 'best_model_path', None)
        return best_path

    def _save_state_dict(self, model: L.LightningModule, artifacts_dir: Path) -> None:
        """Save model state dict."""
        state_dict_path = artifacts_dir / "model_state_dict.pt"
        torch.save(model.state_dict(), state_dict_path)

    def _save_full_model(self, model: L.LightningModule, artifacts_dir: Path) -> None:
        """Save full model."""
        full_model_path = artifacts_dir / "full_model.pt"
        torch.save(model, full_model_path)

    def _save_onnx_model(self, model: L.LightningModule, artifacts_dir: Path) -> None:
        """Save model in ONNX format."""
        try:
            import onnx
        except ImportError:
            print("⚠️ ONNX export skipped: 'onnx' package not installed.")
            print("   To enable ONNX export, install with: pip install onnx onnxruntime")
            print("   Or update dependencies with: uv sync")
            return

        try:
            onnx_path = artifacts_dir / "model.onnx"

            # Create dummy input on the same device as the model
            device = next(model.parameters()).device
            dummy_input = torch.randn(1, 3, 224, 224, device=device)

            # Set model to eval mode for export
            model.eval()

            torch.onnx.export(
                model,
                (dummy_input,),  # inputs as tuple
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            print(f"✅ ONNX model saved to {onnx_path}")
        except Exception as e:
            print(f"❌ Failed to export ONNX model: {e}")
            print("   This is optional - training artifacts will still be saved.")

    def _create_wandb_artifact(self, artifacts_dir: Path, trainer: L.Trainer, checkpoint_path: str) -> None:
        """Create and log W&B artifact with proper ML naming."""
        # Use semantic artifact naming: dataset-model-version
        metadata = self.experiment_manager.get_run_metadata()
        artifact_name = f"{metadata['dataset']}-{metadata['model_architecture']}-{self.experiment_manager.version}"

        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=f"Banana ripeness classifier using {metadata['model_architecture']} - {self.experiment_manager.run_name}",
            metadata={
                "framework": "pytorch-lightning",
                "experiment": self.experiment_manager.experiment_name,
                "run_name": self.experiment_manager.run_name,
                "version": self.experiment_manager.version,
                "architecture": metadata['model_architecture'],
                "dataset": metadata['dataset'],
                "num_classes": metadata['num_classes'],
                "best_val_accuracy": float(trainer.callback_metrics.get("val/accuracy", 0)),
                "best_val_f1": float(trainer.callback_metrics.get("val/f1", 0)),
                "epochs_trained": trainer.current_epoch,
                "hyperparameters": metadata['hyperparameters'],
                "created_at": metadata['created_at'],
                "tags": self.experiment_manager.get_run_tags(),
            }
        )

        # Add files to artifact
        for file_path in artifacts_dir.glob("*"):
            if file_path.is_file():
                artifact.add_file(str(file_path))

        # Add checkpoint
        if os.path.exists(checkpoint_path):
            artifact.add_file(checkpoint_path, name="best_checkpoint.ckpt")

        # Log artifact
        wandb.log_artifact(artifact)
        print(f"✅ Model artifacts logged to W&B: {artifact_name}")


class TrainingOrchestrator:
    """Orchestrates the entire training process following SOLID principles."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        ConfigValidator.validate_config(cfg)

        self.experiment_manager = ExperimentManager(cfg)
        self.component_factory = ComponentFactory()
        self.artifact_manager = ModelArtifactManager(self.experiment_manager, cfg)

    def run_training(self) -> Tuple[L.LightningModule, L.Trainer]:
        """Execute complete training pipeline."""
        print(f"🚀 Starting experiment: {self.experiment_manager.experiment_name}")
        print(f"📊 Run: {self.experiment_manager.run_name}")
        print(f"🏷️  Version: {self.experiment_manager.version}")

        # Set random seed
        self._set_seed()

        # Create components
        datamodule = self._setup_data()

        model = self._setup_model(datamodule)
        logger = self._setup_logger()
        callbacks = self._setup_callbacks()
        trainer = self._setup_trainer(logger, callbacks)

        # Log experiment info
        self._log_experiment_info(model, datamodule)

        # Train model
        self._train_model(trainer, model, datamodule)

        # Test model
        self._test_model(trainer, model, datamodule)

        # Save artifacts
        self._save_artifacts(model, trainer)

        print("✅ Training completed successfully!")
        return model, trainer

    def _set_seed(self) -> None:
        """Set random seed for reproducibility."""
        if 'seed' in self.cfg:
            L.seed_everything(self.cfg.seed, workers=True)

    def _setup_data(self) -> L.LightningDataModule:
        """Setup data module."""
        print("📊 Setting up data module...")
        datamodule = self.component_factory.create_datamodule(self.cfg)
        datamodule.setup(stage=None)

        # Print dataset info
        if hasattr(datamodule, 'print_dataset_info'):
            datamodule.print_dataset_info()

        return datamodule

    def _setup_model(self, datamodule: L.LightningDataModule) -> L.LightningModule:
        """Setup model with optional class weights."""
        print(f"Creating {self.cfg.model.model_name} model...")

        # Get class weights if enabled
        class_weights = None
        if self.cfg.data.get('use_class_weights', False):
            if hasattr(datamodule, 'get_class_weights'):
                class_weights = datamodule.get_class_weights()
                print(f"⚖️ Using class weights: {class_weights}")

        model = self.component_factory.create_model(self.cfg, class_weights)

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created with {total_params:,} total parameters")
        print(f"Trainable parameters: {trainable_params:,}")

        return model

    def _setup_logger(self) -> Logger:
        """Setup logger."""
        print("📝 Setting up logger...")
        return self.component_factory.create_logger(self.cfg, self.experiment_manager)

    def _setup_callbacks(self) -> List[Callback]:
        """Setup callbacks."""
        print("🔧 Setting up callbacks...")
        return self.component_factory.create_callbacks(self.cfg, self.experiment_manager)

    def _setup_trainer(self, logger: Logger, callbacks: List[Callback]) -> L.Trainer:
        """Setup trainer."""
        print("🏃‍♂️ Setting up trainer...")
        return self.component_factory.create_trainer(self.cfg, logger, callbacks)

    def _log_experiment_info(self, model: L.LightningModule, datamodule: L.LightningDataModule) -> None:
        """Log experiment information."""
        if hasattr(model.logger, 'experiment') and hasattr(model.logger.experiment, 'config'):
            # Log comprehensive metadata
            metadata = self.experiment_manager.get_run_metadata()
            model.logger.experiment.config.update(metadata)
            model.logger.experiment.config.update(dict(self.cfg))

    def _train_model(self, trainer: L.Trainer, model: L.LightningModule, datamodule: L.LightningDataModule) -> None:
        """Train the model."""
        print(f"🚀 Starting training for {self.cfg.trainer.max_epochs} epochs...")
        trainer.fit(model, datamodule)

    def _test_model(self, trainer: L.Trainer, model: L.LightningModule, datamodule: L.LightningDataModule) -> None:
        """Test the model."""
        print("🧪 Testing best model...")
        trainer.test(model, datamodule, ckpt_path="best")

    def _save_artifacts(self, model: L.LightningModule, trainer: L.Trainer) -> None:
        """Save model artifacts."""
        print("💾 Saving model artifacts...")
        self.artifact_manager.save_model_artifacts(model, trainer)


# Public API functions
def create_training_orchestrator(cfg: DictConfig) -> TrainingOrchestrator:
    """Create training orchestrator from configuration."""
    return TrainingOrchestrator(cfg)


def run_experiment(cfg: DictConfig) -> Tuple[L.LightningModule, L.Trainer]:
    """Run complete training experiment."""
    orchestrator = create_training_orchestrator(cfg)
    return orchestrator.run_training()
