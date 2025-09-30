"""
Trainer helper utilities for banana ripeness classification.
Implements SOLID principles for clean and maintainable training pipeline.
"""

import os
import torch
import wandb
import lightning as L
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from omegaconf import DictConfig
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import Callback
import hydra
from hydra.utils import instantiate


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
    """Manages experiment metadata and versioning."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.experiment_name = self._generate_experiment_name()
        self.output_dir = self._setup_output_directory()

    def _generate_experiment_name(self) -> str:
        """Generate unique experiment name."""
        if self.cfg.experiment.name and self.cfg.experiment.version:
            return f"{self.cfg.experiment.name}_v{self.cfg.experiment.version}"

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.cfg.model.get('model_name', 'unknown')
        return f"{model_name}_{timestamp}"

    def _setup_output_directory(self) -> Path:
        """Setup experiment output directory."""
        output_dir = Path(f"outputs/{self.experiment_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        (output_dir / "logs").mkdir(exist_ok=True)
        (output_dir / "artifacts").mkdir(exist_ok=True)

        return output_dir

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
        """Create logger from configuration."""
        logger_cfg = cfg.logger.copy()

        # Set experiment metadata
        if hasattr(logger_cfg, 'name') and logger_cfg.name is None:
            logger_cfg.name = experiment_manager.experiment_name

        if hasattr(logger_cfg, 'save_dir'):
            logger_cfg.save_dir = experiment_manager.get_logs_dir()

        if hasattr(logger_cfg, 'tags') and logger_cfg.tags is None:
            logger_cfg.tags = cfg.experiment.get('tags', [])

        if hasattr(logger_cfg, 'notes') and logger_cfg.notes is None:
            logger_cfg.notes = cfg.experiment.get('notes', '')

        return instantiate(logger_cfg)

    @staticmethod
    def create_callbacks(cfg: DictConfig, experiment_manager: ExperimentManager) -> List[Callback]:
        """Create callbacks from configuration."""
        callbacks = []

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

        # Synchroniser automatiquement default_root_dir avec le working dir d'Hydra
        from hydra.core.hydra_config import HydraConfig
        try:
            hydra_cfg = HydraConfig.get()
            hydra_output_dir = hydra_cfg.runtime.output_dir
            trainer_cfg.default_root_dir = hydra_output_dir
            print(f"🔗 Trainer output dir synchronized with Hydra: {hydra_output_dir}")
        except Exception:
            # Fallback si Hydra n'est pas disponible
            pass

        return instantiate(trainer_cfg, logger=logger, callbacks=callbacks)


class ModelArtifactManager:
    """Manages model artifacts and W&B integration."""

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

        # Create W&B artifact
        self._create_wandb_artifact(artifacts_dir, trainer, best_checkpoint_path)

    def _is_wandb_enabled(self) -> bool:
        """Check if W&B logging is enabled."""
        return (hasattr(self.cfg.logger, '_target_') and
                'WandbLogger' in self.cfg.logger._target_)

    def _get_best_checkpoint(self, trainer: L.Trainer) -> Optional[str]:
        """Get path to best checkpoint."""
        if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
            return trainer.checkpoint_callback.best_model_path
        return None

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
                dummy_input,
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
        """Create and log W&B artifact."""
        artifact_name = f"{self.cfg.model.model_name}-banana-classifier"

        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=f"Banana ripeness classifier using {self.cfg.model.model_name}",
            metadata={
                "framework": "pytorch-lightning",
                "architecture": self.cfg.model.model_name,
                "num_classes": self.cfg.model.num_classes,
                "best_val_acc": float(trainer.callback_metrics.get("val/acc", 0)),
                "best_val_f1": float(trainer.callback_metrics.get("val/f1", 0)),
                "epochs_trained": trainer.current_epoch,
                "config": dict(self.cfg),
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
        print(f"Model artifacts logged to W&B: {artifact_name}")


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
        datamodule.setup()

        # Print dataset info
        if hasattr(datamodule, 'print_dataset_info'):
            datamodule.print_dataset_info()

        return datamodule

    def _setup_model(self, datamodule: L.LightningDataModule) -> L.LightningModule:
        """Setup model with optional class weights."""
        print(f"🏗️ Creating {self.cfg.model.model_name} model...")

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
