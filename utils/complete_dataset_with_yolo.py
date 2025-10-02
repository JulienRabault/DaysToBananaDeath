#!/usr/bin/env python3
"""
Script pour compléter le dataset de classification des bananes avec les données YOLO.
Lit les annotations YOLO pour déterminer les classes et copie les images vers les bons dossiers.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import random
from tqdm import tqdm


def read_yolo_annotation(annotation_path: Path) -> List[int]:
    """Lit un fichier d'annotation YOLO et retourne les classes détectées."""
    classes = []
    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # La première valeur est l'ID de classe
                    class_id = int(line.split()[0])
                    classes.append(class_id)
    except Exception as e:
        print(f"Erreur lecture {annotation_path}: {e}")
    return classes


def map_yolo_class_to_folder(yolo_class_id: int) -> str:
    """Mappe les classes YOLO vers les dossiers de classification."""
    # Classe 0: Raw_Banana -> unripe
    # Classe 1: Ripe_Banana -> ripe
    mapping = {
        0: "unripe",    # Raw_Banana
        1: "ripe"       # Ripe_Banana
    }
    return mapping.get(yolo_class_id, "unknowns")


def get_yolo_images_with_classes(yolo_split_path: Path) -> Dict[str, str]:
    """
    Récupère toutes les images YOLO avec leurs classes associées.
    Retourne un dictionnaire {nom_image: classe_destination}
    """
    images_dir = yolo_split_path / "images"
    labels_dir = yolo_split_path / "labels"

    image_class_map = {}

    if not images_dir.exists() or not labels_dir.exists():
        print(f"Dossiers manquants dans {yolo_split_path}")
        return image_class_map

    # Extensions d'images supportées
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    # Parcourir toutes les images
    for img_file in images_dir.iterdir():
        if img_file.suffix in image_extensions:
            # Chercher le fichier d'annotation correspondant
            annotation_file = labels_dir / f"{img_file.stem}.txt"

            if annotation_file.exists() and annotation_file.name != "classes.txt":
                # Lire les classes dans l'annotation
                classes = read_yolo_annotation(annotation_file)

                if classes:
                    # Prendre la première classe détectée (ou la plus fréquente si plusieurs)
                    primary_class = max(set(classes), key=classes.count)
                    destination_folder = map_yolo_class_to_folder(primary_class)
                    image_class_map[str(img_file)] = destination_folder
                else:
                    # Aucune annotation valide -> unknowns
                    image_class_map[str(img_file)] = "unknowns"
            else:
                # Pas d'annotation -> unknowns
                image_class_map[str(img_file)] = "unknowns"

    return image_class_map


def create_directory(path: Path):
    """Crée un dossier s'il n'existe pas."""
    path.mkdir(parents=True, exist_ok=True)


def copy_image_to_class_folder(src_image_path: str, dest_base: Path, class_folder: str):
    """Copie une image vers le dossier de classe approprié."""
    src_path = Path(src_image_path)
    dest_dir = dest_base / class_folder
    create_directory(dest_dir)

    # Nom de destination
    dest_path = dest_dir / src_path.name

    # Gérer les doublons en ajoutant un suffixe
    counter = 1
    original_dest = dest_path
    while dest_path.exists():
        stem = original_dest.stem
        suffix = original_dest.suffix
        dest_path = dest_dir / f"{stem}_yolo_{counter}{suffix}"
        counter += 1

    try:
        shutil.copy2(src_path, dest_path)
        return True
    except Exception as e:
        print(f"Erreur copie {src_path} -> {dest_path}: {e}")
        return False


def process_yolo_split(yolo_split_path: Path, dest_split_path: Path, split_name: str):
    """Traite un split YOLO (train/val/test) et copie vers le dataset de classification."""
    print(f"\n=== Traitement du split {split_name} ===")

    # Récupérer les images avec leurs classes
    images_with_classes = get_yolo_images_with_classes(yolo_split_path)

    if not images_with_classes:
        print(f"Aucune image trouvée dans {yolo_split_path}")
        return

    print(f"Trouvé {len(images_with_classes)} images à traiter")

    # Statistiques par classe
    class_counts = {}
    for class_folder in images_with_classes.values():
        class_counts[class_folder] = class_counts.get(class_folder, 0) + 1

    print("Répartition par classe:")
    for class_name, count in class_counts.items():
        print(f"  - {class_name}: {count} images")

    # Copier les images
    success_count = 0
    for img_path, class_folder in tqdm(images_with_classes.items(), desc=f"Copie {split_name}"):
        if copy_image_to_class_folder(img_path, dest_split_path, class_folder):
            success_count += 1

    print(f"✓ {success_count}/{len(images_with_classes)} images copiées avec succès")


def main():
    # Chemins
    yolo_base = Path("/mnt/data/WORK/DATA/banana_yolo")
    dest_base = Path("/mnt/data/WORK/DATA")

    # Vérifier que le dataset YOLO existe
    if not yolo_base.exists():
        print(f"Erreur : {yolo_base} n'existe pas")
        return

    print("=== Début de l'intégration des données YOLO ===")
    print(f"Source YOLO : {yolo_base}")
    print(f"Destination : {dest_base}")

    # Mapping des splits
    yolo_splits = {
        "train": yolo_base / "train",
        "val": yolo_base / "val",  # val YOLO -> valid classification
        "test": yolo_base / "test"
    }

    dest_splits = {
        "train": dest_base / "train",
        "val": dest_base / "valid",  # Attention: val -> valid
        "test": dest_base / "test"
    }

    # Traiter chaque split
    for split_name, yolo_split_path in yolo_splits.items():
        if yolo_split_path.exists():
            dest_split_path = dest_splits[split_name]
            process_yolo_split(yolo_split_path, dest_split_path, split_name)
        else:
            print(f"Split {split_name} non trouvé : {yolo_split_path}")

    print("\n=== Intégration terminée ===")

    # Afficher un résumé final
    print("\nRésumé des dossiers de destination:")
    for split_name, dest_path in dest_splits.items():
        if dest_path.exists():
            print(f"\n{split_name.upper()} ({dest_path}):")
            for class_dir in sorted(dest_path.iterdir()):
                if class_dir.is_dir():
                    count = len(list(class_dir.glob("*")))
                    print(f"  - {class_dir.name}: {count} images")


if __name__ == "__main__":
    # Fixer la seed pour la reproductibilité
    random.seed(42)
    main()
