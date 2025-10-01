#!/usr/bin/env python3
"""
Inference script for banana ripeness classification.
Loads a trained model from checkpoints or W&B and performs predictions.
Designed for web deployment and FastAPI integration.
Supports only ResNet50 and ViT-B/16.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional
import json
import logging
import tempfile
import os
from omegaconf import OmegaConf, DictConfig

# Import supported models only
from model import ResNetClassifier, VisionTransformerClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BananaClassifierInference:
    """Inference class for banana ripeness classification with W&B integration."""

    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 wandb_run_path: Optional[str] = None,
                 wandb_artifact_name: Optional[str] = None,
                 config_path: Optional[str] = None,
                 model_type: str = "resnet50",
                 device: str = "auto",
                 img_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the inference class.

        Args:
            checkpoint_path: Path to local model checkpoint (.ckpt)
            wandb_run_path: W&B run path (e.g., "username/project/run_id")
            wandb_artifact_name: W&B artifact name for model
            config_path: Path to config file (YAML)
            model_type: Type of model ("resnet50" or "vit_b_16")
            device: Device to run inference on ("cpu", "cuda", "auto")
            img_size: Image size for preprocessing (height, width)
        """
        self.device = self._setup_device(device)

        # Class names (consistent with training)
        self.class_names = ['overripe', 'ripe', 'rotten', 'unripe']
        self.num_classes = len(self.class_names)

        # Load configuration if provided
        self.config = self._load_config(config_path) if config_path else None

        # Override parameters from config if available
        if self.config:
            model_type = self.config.model.get('model_name', model_type)
            img_size = tuple(self.config.data.get('img_size', img_size))

        self.model_type = model_type
        self.img_size = img_size

        # Load model from various sources
        if checkpoint_path:
            self.checkpoint_path = checkpoint_path
            self.model = self._load_model_from_checkpoint(checkpoint_path)
        elif wandb_run_path or wandb_artifact_name:
            self.model = self._load_model_from_wandb(wandb_run_path, wandb_artifact_name)
        else:
            raise ValueError("Must provide either checkpoint_path, wandb_run_path, or wandb_artifact_name")

        # Setup preprocessing
        self.transform = self._setup_transform()

        logger.info(f"Inference setup complete. Model: {model_type}, Device: {self.device}")

    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("Using CPU for inference")
        return torch.device(device)

    def _load_config(self, config_path: str) -> DictConfig:
        """Load Hydra configuration from file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        logger.info(f"Loading config from {config_path}")
        return OmegaConf.load(config_path)

    def _load_model_from_checkpoint(self, checkpoint_path: str) -> torch.nn.Module:
        """Load model from local checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading model from checkpoint: {checkpoint_path}")

        # Map model type to class (restricted)
        model_classes = {
            "resnet50": ResNetClassifier,
            "vit_b_16": VisionTransformerClassifier,
        }

        if self.model_type not in model_classes:
            raise ValueError(f"Unsupported model type: {self.model_type}. Only 'resnet50' and 'vit_b_16' are supported.")

        model_class = model_classes[self.model_type]

        # Load from checkpoint
        model = model_class.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device
        )

        # Set to evaluation mode
        model.eval()
        model.to(self.device)

        return model

    def _load_model_from_wandb(self, run_path: Optional[str] = None,
                              artifact_name: Optional[str] = None) -> torch.nn.Module:
        """Load model from W&B run or artifact."""
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb is required to load models from W&B. Install with: pip install wandb")

        if run_path:
            logger.info(f"Loading model from W&B run: {run_path}")

            # Initialize wandb in offline mode for inference
            wandb.init(mode="offline")

            # Get the run
            api = wandb.Api()
            run = api.run(run_path)

            # Download the best checkpoint
            checkpoint_files = [f for f in run.files() if f.name.endswith('.ckpt')]
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint files found in run {run_path}")

            # Find the best checkpoint (usually contains "best" or highest val_acc)
            best_checkpoint = None
            best_score = -1
            for file in checkpoint_files:
                if 'best' in file.name or 'val_acc' in file.name:
                    # Try to extract validation accuracy from filename
                    try:
                        if 'val_acc' in file.name:
                            score = float(file.name.split('val_acc_')[1].split('.ckpt')[0])
                            if score > best_score:
                                best_score = score
                                best_checkpoint = file
                    except:
                        pass

            if best_checkpoint is None:
                best_checkpoint = checkpoint_files[0]  # Fallback to first checkpoint

            logger.info(f"Downloading checkpoint: {best_checkpoint.name}")

            # Download to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_path = os.path.join(temp_dir, best_checkpoint.name)
                best_checkpoint.download(temp_dir)

                # Load the model
                model = self._load_model_from_checkpoint(checkpoint_path)

            wandb.finish()
            return model

        elif artifact_name:
            logger.info(f"Loading model from W&B artifact: {artifact_name}")

            # Initialize wandb
            wandb.init(mode="offline")

            # Download artifact
            artifact = wandb.use_artifact(artifact_name)
            artifact_dir = artifact.download()

            # Find checkpoint in artifact
            checkpoint_files = list(Path(artifact_dir).glob("*.ckpt"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint files found in artifact {artifact_name}")

            checkpoint_path = checkpoint_files[0]  # Use first checkpoint found
            logger.info(f"Using checkpoint from artifact: {checkpoint_path}")

            model = self._load_model_from_checkpoint(str(checkpoint_path))

            wandb.finish()
            return model

        else:
            raise ValueError("Must provide either run_path or artifact_name for W&B loading")

    def _setup_transform(self) -> A.Compose:
        """Setup image preprocessing transforms."""
        # Use same normalization as training (ImageNet stats)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        return A.Compose([
            A.Resize(height=self.img_size[0], width=self.img_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess a single image for inference.

        Args:
            image: Can be a file path, PIL Image, or numpy array

        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert to PIL Image if needed
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Convert to numpy for albumentations
        image_np = np.array(image)

        # Apply transforms
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image']

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    def predict_single(self, image: Union[str, Path, Image.Image, np.ndarray],
                      return_top_k: int = 4) -> Dict:
        """
        Predict ripeness for a single image.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            return_top_k: Number of top predictions to return

        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)

            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, k=min(return_top_k, self.num_classes))

            predicted_class_idx = top_indices[0][0].item()
            confidence = top_probs[0][0].item()

        # Prepare results
        predicted_class = self.class_names[predicted_class_idx]

        # All class probabilities
        all_probabilities = {
            class_name: probabilities[0][i].item()
            for i, class_name in enumerate(self.class_names)
        }

        # Top-k predictions
        top_predictions = [
            {
                "class": self.class_names[top_indices[0][i].item()],
                "probability": top_probs[0][i].item(),
                "index": top_indices[0][i].item()
            }
            for i in range(top_probs.size(1))
        ]

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "predicted_index": predicted_class_idx,
            "class_probabilities": all_probabilities,
            "top_predictions": top_predictions,
            "model_info": {
                "model_type": self.model_type,
                "image_size": self.img_size,
                "device": str(self.device)
            }
        }

    def predict_batch(self, images: List[Union[str, Path, Image.Image, np.ndarray]]) -> List[Dict]:
        """
        Predict ripeness for multiple images.

        Args:
            images: List of input images

        Returns:
            List of prediction dictionaries
        """
        results = []
        for i, image in enumerate(images):
            try:
                result = self.predict_single(image)
                result["batch_index"] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                results.append({
                    "batch_index": i,
                    "error": str(e),
                    "predicted_class": None,
                    "confidence": 0.0,
                    "class_probabilities": {},
                    "predicted_index": -1
                })
        return results

    def predict_directory(self, image_dir: Union[str, Path],
                         output_file: Optional[str] = None,
                         recursive: bool = False) -> List[Dict]:
        """
        Predict ripeness for all images in a directory.

        Args:
            image_dir: Directory containing images
            output_file: Optional file to save results as JSON
            recursive: Whether to search subdirectories

        Returns:
            List of prediction dictionaries with filenames
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {image_dir}")

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        if recursive:
            image_files = [
                f for f in image_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
        else:
            image_files = [
                f for f in image_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]

        if not image_files:
            logger.warning(f"No image files found in {image_dir}")
            return []

        logger.info(f"Processing {len(image_files)} images from {image_dir}")

        results = []
        for image_file in image_files:
            try:
                prediction = self.predict_single(image_file)
                prediction["filename"] = image_file.name
                prediction["filepath"] = str(image_file.relative_to(image_dir))
                prediction["full_path"] = str(image_file)
                results.append(prediction)

                logger.info(f"{image_file.name}: {prediction['predicted_class']} "
                           f"({prediction['confidence']:.3f})")

            except Exception as e:
                logger.error(f"Error processing {image_file.name}: {e}")
                results.append({
                    "filename": image_file.name,
                    "filepath": str(image_file.relative_to(image_dir)),
                    "full_path": str(image_file),
                    "error": str(e),
                    "predicted_class": None,
                    "confidence": 0.0,
                    "class_probabilities": {},
                    "predicted_index": -1
                })

        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {output_file}")

        return results

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "image_size": self.img_size,
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "has_cuda": torch.cuda.is_available(),
            "pytorch_version": torch.__version__
        }


def create_inference_from_config(config_path: str,
                               checkpoint_path: Optional[str] = None,
                               wandb_run_path: Optional[str] = None) -> BananaClassifierInference:
    """
    Create inference object from a Hydra config file.

    Args:
        config_path: Path to the Hydra config file
        checkpoint_path: Path to model checkpoint (optional)
        wandb_run_path: W&B run path (optional)

    Returns:
        Configured inference object
    """
    return BananaClassifierInference(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        wandb_run_path=wandb_run_path
    )


def find_best_checkpoint(checkpoint_dir: Union[str, Path]) -> str:
    """
    Find the best checkpoint in a directory based on validation accuracy.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to the best checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    # Try to find checkpoint with highest val_acc
    best_checkpoint = None
    best_score = -1

    for checkpoint_file in checkpoint_files:
        if 'val_acc' in checkpoint_file.name:
            try:
                score = float(checkpoint_file.name.split('val_acc_')[1].split('.ckpt')[0])
                if score > best_score:
                    best_score = score
                    best_checkpoint = checkpoint_file
            except:
                pass

    if best_checkpoint is None:
        # Fallback to last.ckpt or first checkpoint
        last_checkpoint = checkpoint_dir / "last.ckpt"
        if last_checkpoint.exists():
            best_checkpoint = last_checkpoint
        else:
            best_checkpoint = checkpoint_files[0]

    logger.info(f"Selected checkpoint: {best_checkpoint} (score: {best_score if best_score > -1 else 'unknown'})")
    return str(best_checkpoint)


def main():
    """Example usage of the inference system."""
    import argparse

    parser = argparse.ArgumentParser(description="Banana Ripeness Classification Inference")
    parser.add_argument("--checkpoint", help="Path to model checkpoint")
    parser.add_argument("--checkpoint_dir", help="Directory containing checkpoints (auto-select best)")
    parser.add_argument("--wandb_run", help="W&B run path (e.g., username/project/run_id)")
    parser.add_argument("--wandb_artifact", help="W&B artifact name")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--image", help="Path to single image")
    parser.add_argument("--directory", help="Path to directory of images")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--model_type", default="resnet50", help="Model type ('resnet50' or 'vit_b_16')")
    parser.add_argument("--device", default="auto", help="Device (cpu/cuda/auto)")

    args = parser.parse_args()

    # Determine checkpoint source
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.checkpoint_dir:
        checkpoint_path = find_best_checkpoint(args.checkpoint_dir)

    # Create inference object
    classifier = BananaClassifierInference(
        checkpoint_path=checkpoint_path,
        wandb_run_path=args.wandb_run,
        wandb_artifact_name=args.wandb_artifact,
        config_path=args.config,
        model_type=args.model_type,
        device=args.device
    )

    # Print model info
    model_info = classifier.get_model_info()
    print(f"\n🍌 Banana Classifier Loaded:")
    print(f"   Model: {model_info['model_type']}")
    print(f"   Classes: {', '.join(model_info['class_names'])}")
    print(f"   Parameters: {model_info['total_parameters']:,}")
    print(f"   Device: {model_info['device']}")
    print(f"   Image Size: {model_info['image_size']}")

    # Run inference
    if args.image:
        print(f"\n🔍 Analyzing single image: {args.image}")
        result = classifier.predict_single(args.image)
        print(f"   Prediction: {result['predicted_class']} ({result['confidence']:.3f})")
        print(f"   Top predictions:")
        for i, pred in enumerate(result['top_predictions'][:3]):
            print(f"      {i+1}. {pred['class']}: {pred['probability']:.3f}")

    elif args.directory:
        print(f"\n🔍 Analyzing directory: {args.directory}")
        results = classifier.predict_directory(args.directory, args.output, args.recursive)

        # Summary statistics
        predictions = [r['predicted_class'] for r in results if r['predicted_class']]
        if predictions:
            from collections import Counter
            summary = Counter(predictions)
            print(f"\n📊 Summary ({len(predictions)} images):")
            for class_name, count in summary.items():
                print(f"   {class_name}: {count} ({count/len(predictions)*100:.1f}%)")

    else:
        print("Please specify either --image or --directory")
        print("\nExample usage:")
        print("  # Local checkpoint")
        print("  python inference.py --checkpoint path/to/model.ckpt --image banana.jpg")
        print("  # Auto-select best checkpoint from directory")
        print("  python inference.py --checkpoint_dir outputs/experiment/checkpoints --directory test_images")
        print("  # Load from W&B")
        print("  python inference.py --wandb_run username/project/run_id --image banana.jpg")


if __name__ == "__main__":
    main()
