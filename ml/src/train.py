#!/usr/bin/env python3
"""
Training script for banana ripeness classification.
Uses Hydra for configuration management and follows SOLID principles.
"""

import hydra
from omegaconf import DictConfig
from trainer_helper import run_experiment
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)



@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training function."""
    try:
        # Run the complete training experiment
        model, trainer = run_experiment(cfg)

    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    train()
