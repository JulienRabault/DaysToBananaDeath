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

        # Always add a timestamp + short random suffix to ensure uniqueness even within the same second
        return unique_path(
            base_run_name,
            suffix_format="_{timestamp}_{rand:6}",
            if_exists_only=False,
            return_str=True,
        )

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
        # (unique_output_dir / "metrics").mkdir(exist_ok=True)
        # (unique_output_dir / "plots").mkdir(exist_ok=True)

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

        # Ensure we never resume an old run implicitly
        os.environ["WANDB_RESUME"] = "never"
        # Clear any preset run identifiers that could force resume
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ.pop("WANDB_RUN_NAME", None)

        # Use the run_name (descriptive) for WandB
        logger_cfg.name = experiment_manager.run_name
        logger_cfg.save_dir = experiment_manager.get_logs_dir()

        # Set standardized tags and metadata
        logger_cfg.tags = experiment_manager.get_run_tags()

        # Set experiment group for organization
        if hasattr(logger_cfg, 'group'):
            logger_cfg.group = experiment_manager.experiment_name

        # Do NOT set WandB "version" here because WandbLogger may treat it as the run id
        if hasattr(logger_cfg, 'version'):
            # Remove/ignore any provided version to avoid collisions
            try:
                del logger_cfg["version"]
            except Exception:
                logger_cfg.version = None

        # Force a new unique run id every time
        import uuid
        logger_cfg.id = str(uuid.uuid4())

        # Explicitly disable resume behavior
        logger_cfg.resume = "never"

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
    """Manages model artifacts with simplified approach."""

    def __init__(self, experiment_manager: ExperimentManager, cfg: DictConfig):
        self.experiment_manager = experiment_manager
        self.cfg = cfg

    def save_model_artifacts(self, model: L.LightningModule, trainer: L.Trainer) -> None:
        """Save model artifacts - local config + WandB ONNX artifact."""
        artifacts_dir = Path(self.experiment_manager.get_artifacts_dir())

        # Toujours sauvegarder la config localement
        self._save_config(artifacts_dir)

        # Sauvegarder ONNX localement et comme artifact WandB
        best_checkpoint_path = self._get_best_checkpoint(trainer)
        if best_checkpoint_path:
            # Sauvegarder ONNX localement
            onnx_path = self._save_onnx_model(model, best_checkpoint_path, artifacts_dir)

            # Si WandB est activé, créer un artifact ONNX unique pour ce run
            if self._is_wandb_enabled() and onnx_path and onnx_path.exists():
                self._create_onnx_wandb_artifact(trainer, onnx_path)

    def _save_config(self, artifacts_dir: Path) -> None:
        """Save experiment configuration to artifacts directory."""
        import yaml
        config_path = artifacts_dir / "config.yaml"

        # Convert DictConfig to regular dict for serialization
        config_dict = dict(self.cfg)

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        print(f"✅ Configuration saved to {config_path}")

    def _is_wandb_enabled(self) -> bool:
        """Check if W&B logging is enabled."""
        return (hasattr(self.cfg.logger, '_target_') and
                'WandbLogger' in self.cfg.logger._target_)

    def _get_best_checkpoint(self, trainer: L.Trainer) -> Optional[str]:
        """Get path to best checkpoint."""
        best_path: Optional[str] = None
        for cb in getattr(trainer, 'callbacks', []) or []:
            if isinstance(cb, ModelCheckpoint):
                if getattr(cb, 'best_model_path', None):
                    best_path = cb.best_model_path
                    break
        if not best_path and hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
            best_path = getattr(trainer.checkpoint_callback, 'best_model_path', None)
        return best_path

    def _save_onnx_model(self, model: L.LightningModule, checkpoint_path: str, artifacts_dir: Path) -> Optional[Path]:
        """Save model in ONNX format."""
        try:
            import onnx
        except ImportError:
            print("⚠️ ONNX export skipped: 'onnx' package not installed.")
            return None

        try:
            # Charger le meilleur modèle depuis le checkpoint
            best_model = type(model).load_from_checkpoint(checkpoint_path)
            best_model.eval()

            onnx_path = artifacts_dir / "model.onnx"

            # Create dummy input on the same device as the model
            device = next(best_model.parameters()).device
            dummy_input = torch.randn(1, 3, 224, 224, device=device)

            torch.onnx.export(
                best_model,
                (dummy_input,),
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            print(f"✅ ONNX model saved to {onnx_path}")
            return onnx_path
        except Exception as e:
            print(f"❌ Failed to export ONNX model: {e}")
            return None

    def _create_onnx_wandb_artifact(self, trainer: L.Trainer, onnx_path: Path) -> None:
        """Create WandB artifact for ONNX model - shared 'onnx' artifact for easy deployment."""
        try:
            # Vérifier que le logger WandB est disponible
            if not hasattr(trainer, 'logger') or trainer.logger is None:
                print("⚠️ Pas de logger disponible")
                return

            if not hasattr(trainer.logger, 'experiment') or trainer.logger.experiment is None:
                print("⚠️ Logger WandB non disponible")
                return

            # Récupérer les infos du run
            run_name = trainer.logger.experiment.name
            metric_value = trainer.callback_metrics.get("val/accuracy", 0)
            current_accuracy = float(metric_value) if metric_value is not None else 0.0

            # Nom d'artifact COMMUN pour le déploiement
            artifact_name = "onnx"

            # Récupérer le meilleur modèle existant pour comparaison
            best_accuracy_so_far = self._get_current_best_accuracy(trainer)
            is_best_model = current_accuracy > best_accuracy_so_far

            if is_best_model:
                print(f"   🏆 NOUVEAU MEILLEUR MODÈLE! {current_accuracy:.4f} > {best_accuracy_so_far:.4f}")
            else:
                print(f"   📊 Modèle actuel: {current_accuracy:.4f} (meilleur: {best_accuracy_so_far:.4f})")

            # Créer l'artifact
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"ONNX models for deployment - Latest from: {run_name} (val_accuracy: {current_accuracy:.4f})",
                metadata={
                    "latest_run": run_name,
                    "latest_accuracy": current_accuracy,
                    "best_accuracy": max(current_accuracy, best_accuracy_so_far),
                    "format": "onnx",
                    "framework": "pytorch-lightning",
                    "opset_version": 11,
                    "input_shape": [1, 3, 224, 224],
                    "deployment_ready": True,
                    "updated_at": datetime.now().isoformat(),
                    "is_best_model": is_best_model
                }
            )

            # Ajouter le fichier ONNX avec nom incluant le run pour traçabilité
            artifact.add_file(str(onnx_path), name=f"model-{run_name}.onnx")

            # Définir les aliases pour le déploiement
            aliases = ["latest"]  # Toujours "latest" pour le dernier modèle

            if is_best_model:
                aliases.append("best")
                print(f"   🎯 Modèle marqué comme BEST et remplace l'ancien meilleur")
            else:
                print(f"   📝 Modèle marqué comme LATEST seulement")

            # Logger l'artifact avec les alias appropriés
            trainer.logger.experiment.log_artifact(artifact, aliases=aliases)

            print(f"✅ ONNX WandB artifact updated: {artifact_name}")
            print(f"   Fichier ajouté: model-{run_name}.onnx")
            print(f"   Aliases: {aliases}")
            print(f"   📦 Pour déploiement: utilisez l'artifact 'onnx' avec alias 'best' ou 'latest'")

        except Exception as e:
            print(f"⚠️ Erreur lors de la création de l'artifact WandB: {e}")

    def _get_current_best_accuracy(self, trainer: L.Trainer) -> float:
        """Récupère la meilleure accuracy de l'artifact ONNX existant."""
        try:
            # Essayer de récupérer l'artifact ONNX existant avec tag "best"
            api = wandb.Api()
            project_name = trainer.logger.experiment.project

            try:
                # Chercher l'artifact avec tag "best"
                artifact = api.artifact(f"{project_name}/onnx:best")
                best_accuracy = artifact.metadata.get("best_accuracy", 0.0)
                print(f"   📋 Meilleur modèle existant trouvé: {best_accuracy:.4f}")
                return best_accuracy
            except wandb.errors.CommError:
                # Pas d'artifact "best" existant, essayer "latest"
                try:
                    artifact = api.artifact(f"{project_name}/onnx:latest")
                    latest_accuracy = artifact.metadata.get("latest_accuracy", 0.0)
                    print(f"   📋 Dernier modèle trouvé: {latest_accuracy:.4f}")
                    return latest_accuracy
                except wandb.errors.CommError:
                    print(f"   📋 Aucun artifact ONNX existant, ce sera le premier")
                    return 0.0
        except Exception as e:
            print(f"   ⚠️ Erreur lors de la récupération du meilleur modèle: {e}")
            return 0.0

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
        datamodule.setup(stage="fit")

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
