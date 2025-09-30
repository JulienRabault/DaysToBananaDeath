import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import lightning as L
from typing import Optional, Dict, Tuple, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class BananaRipenessDataset(Dataset):
    """PyTorch Dataset for banana ripeness classification."""

    def __init__(self, data_dir: str, split: str = "train", transform=None):
        """
        Args:
            data_dir: Root directory of the data
            split: 'train', 'valid', or 'test'
            transform: Transformations to apply to images
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # Banana ripeness classes
        self.classes = ['overripe', 'ripe', 'rotten', 'unripe']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Load image paths and their labels
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and their labels for the given split."""
        samples = []
        split_dir = os.path.join(self.data_dir, self.split)

        for class_name in self.classes:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        label = self.class_to_idx[class_name]
                        samples.append((img_path, label))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform:
            if isinstance(self.transform, A.Compose):
                # For albumentations
                image = np.array(image)
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                # For torchvision
                image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Returns the class distribution in this dataset."""
        distribution = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            class_name = self.classes[label]
            distribution[class_name] += 1
        return distribution


class BananaDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for banana dataset."""

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 32,
        num_workers: int = 4,
        img_size: Tuple[int, int] = (224, 224),
        use_augmentation: bool = False,
        pin_memory: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.use_augmentation = use_augmentation
        self.pin_memory = pin_memory

        # Ripeness classes
        self.classes = ['overripe', 'ripe', 'rotten', 'unripe']
        self.num_classes = len(self.classes)

        # ImageNet statistics for normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def get_transforms(self, split: str):
        """Returns appropriate transformations for each split."""
        if split == "train" and self.use_augmentation:
            # Augmentation transformations for training
            return A.Compose([
                A.Resize(height=self.img_size[0], width=self.img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.5
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.ImageCompression(quality_lower=85, quality_upper=100),
                ], p=0.3),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
        else:
            # Simple transformations for validation and test
            return A.Compose([
                A.Resize(height=self.img_size[0], width=self.img_size[1]),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])

    def setup(self, stage: Optional[str] = None):
        """Prepare datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = BananaRipenessDataset(
                self.data_dir,
                split="train",
                transform=self.get_transforms("train")
            )
            self.val_dataset = BananaRipenessDataset(
                self.data_dir,
                split="valid",
                transform=self.get_transforms("valid")
            )

        if stage == "test" or stage is None:
            self.test_dataset = BananaRipenessDataset(
                self.data_dir,
                split="test",
                transform=self.get_transforms("test")
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights to handle class imbalance."""
        if not hasattr(self, 'train_dataset'):
            self.setup('fit')

        distribution = self.train_dataset.get_class_distribution()
        total_samples = sum(distribution.values())

        weights = []
        for class_name in self.classes:
            class_count = distribution[class_name]
            weight = total_samples / (len(self.classes) * class_count) if class_count > 0 else 0
            weights.append(weight)

        return torch.FloatTensor(weights)

    def print_dataset_info(self):
        """Display dataset information."""
        if not hasattr(self, 'train_dataset'):
            self.setup('fit')

        print(f"\n=== Banana Dataset Information ===")
        print(f"Classes: {self.classes}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Image size: {self.img_size}")
        print(f"Batch size: {self.batch_size}")

        print(f"\nTrain Distribution:")
        train_dist = self.train_dataset.get_class_distribution()
        for class_name, count in train_dist.items():
            print(f"  {class_name}: {count}")

        print(f"\nValidation Distribution:")
        val_dist = self.val_dataset.get_class_distribution()
        for class_name, count in val_dist.items():
            print(f"  {class_name}: {count}")

        if hasattr(self, 'test_dataset'):
            print(f"\nTest Distribution:")
            test_dist = self.test_dataset.get_class_distribution()
            for class_name, count in test_dist.items():
                print(f"  {class_name}: {count}")


def create_banana_datamodule(
    data_dir: str = "data",
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (224, 224),
    use_augmentation: bool = False,
    pin_memory: bool = True,
    *args, **kwargs
) -> BananaDataModule:
    """
    Utility function to create and configure the DataModule.

    Args:
        data_dir: Directory containing the data
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        img_size: Image size (height, width)
        use_augmentation: Use data augmentation
        pin_memory: Pin memory for DataLoader
        use_class_weights: Whether to use class weights (passé au trainer_helper)

    Returns:
        Configured BananaDataModule
    """
    return BananaDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        use_augmentation=use_augmentation,
        pin_memory=pin_memory

    )


if __name__ == "__main__":
    # Usage example
    import numpy as np

    # Create DataModule
    dm = create_banana_datamodule(
        data_dir="data",
        batch_size=16,
        num_workers=2,
        img_size=(224, 224)
    )

    # Prepare data
    dm.setup()

    # Display information
    dm.print_dataset_info()

    # Test loading a batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    images, labels = batch

    print(f"\nDataLoader Test:")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels in this batch: {labels.tolist()}")
    print(f"Corresponding classes: {[dm.classes[label] for label in labels]}")
