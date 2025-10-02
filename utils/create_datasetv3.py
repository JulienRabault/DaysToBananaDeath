#!/usr/bin/env python3
"""
Script pour créer le dataset v3 en intégrant les données synthétiques et équilibrer les classes unknowns.

Sources :
- Dataset v2 existant : /mnt/data/WORK/DATA/datasetv2/
- Dataset synthétique : /mnt/data/WORK/DATA/Synthetic Dataset/
- Nouveau dataset : /mnt/data/WORK/DATA/datasetv3/

Mapping des classes synthétiques (Ripeness) :
- Ripeness A -> unripe
- Ripeness B -> ripe
- Ripeness C -> overripe
- Ripeness D -> rotten

Nouveauté v3 :
- Intégration des images synthétiques
- Gestion du dossier "nombre de bananes"
- Équilibrage automatique des classes unknowns
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import re
from tqdm import tqdm
from collections import defaultdict
from PIL import Image, ImageOps
import random
import math

# Support optionnel pour pillow-heif
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False
    print("⚠️  pillow-heif non disponible, support HEIC/HEIF désactivé")


def create_directory(path: Path):
    """Crée un dossier s'il n'existe pas."""
    path.mkdir(parents=True, exist_ok=True)


def convert_and_copy_image(src_path: Path, dest_path: Path, quality: int = 95) -> bool:
    """Convertit une image vers JPG avec qualité optimisée pour l'IA."""
    try:
        dest_path = dest_path.with_suffix('.jpg')

        with Image.open(src_path) as img:
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            img = ImageOps.exif_transpose(img)
            img.save(dest_path, 'JPEG', quality=quality, optimize=True)

        return True

    except Exception as e:
        print(f"    ❌ Erreur conversion {src_path.name}: {e}")
        return False


def normalize_filename(filename: str, class_name: str, split_name: str, counter: int, source: str = "") -> str:
    """Génère un nom de fichier normalisé avec source optionnelle."""
    source_suffix = f"_{source}" if source else ""
    return f"{class_name}_{split_name}_{counter:06d}{source_suffix}.jpg"


def get_synthetic_class_mapping() -> Dict[str, str]:
    """Mapping des classes synthétiques vers nos classes."""
    return {
        "Ripeness A": "unripe",
        "Ripeness B": "ripe",
        "Ripeness C": "overripe",
        "Ripeness D": "rotten",
        # Variantes possibles
        "A": "unripe",
        "B": "ripe",
        "C": "overripe",
        "D": "rotten"
    }


def explore_synthetic_dataset(synthetic_path: Path) -> Dict:
    """Explore et analyse la structure du dataset synthétique."""
    print(f"🔍 Exploration du dataset synthétique: {synthetic_path}")

    structure = {
        "splits": [],
        "classes": [],
        "banana_count_dirs": [],
        "total_images": 0,
        "structure_type": "unknown"
    }

    if not synthetic_path.exists():
        print(f"❌ Dataset synthétique non trouvé: {synthetic_path}")
        return structure

    # Explorer la structure
    for item in synthetic_path.iterdir():
        if item.is_dir():
            print(f"  📁 {item.name}")

            # Chercher des patterns de splits
            if item.name.lower() in ["train", "test", "validation", "valid"]:
                structure["splits"].append(item.name)
                print(f"    ✓ Split détecté: {item.name}")

                # Explorer les sous-dossiers du split
                for subitem in item.iterdir():
                    if subitem.is_dir():
                        print(f"      📁 {subitem.name}")

                        # Détecter les classes de maturité
                        if any(ripeness in subitem.name for ripeness in ["Ripeness", "A", "B", "C", "D"]):
                            if subitem.name not in structure["classes"]:
                                structure["classes"].append(subitem.name)
                                print(f"        ✓ Classe détectée: {subitem.name}")

                        # Détecter les dossiers de nombre de bananes
                        if any(keyword in subitem.name.lower() for keyword in ["banana", "number", "count", "nombre"]):
                            if subitem.name not in structure["banana_count_dirs"]:
                                structure["banana_count_dirs"].append(subitem.name)
                                print(f"        ✓ Dossier nombre de bananes: {subitem.name}")

    # Compter les images totales
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG',
                       '.bmp', '.BMP', '.tiff', '.TIFF', '.webp', '.WEBP'}
    if HEIF_SUPPORT:
        image_extensions.update({'.heic', '.HEIC', '.heif', '.HEIF'})

    for img_file in synthetic_path.rglob("*"):
        if img_file.is_file() and img_file.suffix in image_extensions:
            structure["total_images"] += 1

    print(f"📊 Résumé de l'exploration:")
    print(f"  - Splits trouvés: {structure['splits']}")
    print(f"  - Classes trouvées: {structure['classes']}")
    print(f"  - Dossiers nombre bananes: {structure['banana_count_dirs']}")
    print(f"  - Total images: {structure['total_images']}")

    return structure


def copy_v2_to_v3(v2_path: Path, v3_path: Path) -> bool:
    """Copie le dataset v2 vers v3 comme base."""
    print(f"=== Copie du dataset v2 vers v3 ===")

    if not v2_path.exists():
        print(f"❌ Dataset v2 non trouvé à {v2_path}")
        return False

    if v3_path.exists():
        print(f"Dataset v3 existe déjà à {v3_path}")
        response = input("Voulez-vous le supprimer et recommencer ? (y/N): ")
        if response.lower() == 'y':
            shutil.rmtree(v3_path)
        else:
            print("Arrêt du processus.")
            return False

    print("Copie en cours...")
    shutil.copytree(v2_path, v3_path)
    print(f"✓ Dataset v2 copié vers v3")
    return True


def count_images_by_class(dataset_path: Path) -> Dict[str, Dict[str, int]]:
    """Compte les images par classe et par split."""
    counts = defaultdict(lambda: defaultdict(int))

    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG',
                       '.bmp', '.BMP', '.tiff', '.TIFF', '.webp', '.WEBP'}

    for split_dir in dataset_path.iterdir():
        if split_dir.is_dir() and split_dir.name in ["train", "valid", "test"]:
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    images = [f for f in class_dir.iterdir()
                             if f.is_file() and f.suffix in image_extensions]
                    counts[split_dir.name][class_dir.name] = len(images)

    return dict(counts)


def balance_unknowns_class(v3_path: Path):
    """Équilibre la classe unknowns avec les autres classes dans chaque split."""
    print(f"\n🎯 Équilibrage de la classe unknowns")

    # Compter les images actuelles
    current_counts = count_images_by_class(v3_path)

    for split_name in ["train", "valid", "test"]:
        if split_name not in current_counts:
            continue

        split_path = v3_path / split_name
        unknowns_path = split_path / "unknowns"

        if not unknowns_path.exists():
            create_directory(unknowns_path)
            current_counts[split_name]["unknowns"] = 0

        # Calculer la moyenne des autres classes (hors unknowns)
        other_classes = {k: v for k, v in current_counts[split_name].items() if k != "unknowns"}

        if not other_classes:
            print(f"  ⚠️  Aucune autre classe trouvée dans {split_name}")
            continue

        # Ratio souhaité : unknowns = 25% de la moyenne des autres classes
        # Pour avoir un ratio 20%-20%-20%-20%-10%, unknowns doit être 1/4 des autres
        avg_other_classes = int(sum(other_classes.values()) / len(other_classes))
        target_count = max(5, int(avg_other_classes * 0.25))  # Minimum 5, sinon 25% de la moyenne
        current_unknowns = current_counts[split_name].get("unknowns", 0)

        print(f"\n--- {split_name.upper()} ---")
        print(f"  Autres classes (moyenne): {avg_other_classes}")
        print(f"  Target unknowns (25% = 10% du total): {target_count}")
        print(f"  Unknowns actuels: {current_unknowns}")

        if current_unknowns == target_count:
            print(f"  ✓ Déjà équilibré")
            continue
        elif current_unknowns > target_count:
            # Réduire les unknowns
            to_remove = current_unknowns - target_count
            print(f"  🔄 Réduction de {to_remove} images unknowns")

            # Sélectionner aléatoirement les images à déplacer vers backup
            image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
            all_unknowns = [f for f in unknowns_path.iterdir()
                           if f.is_file() and f.suffix in image_extensions]

            random.shuffle(all_unknowns)
            images_to_remove = all_unknowns[:to_remove]

            # Créer backup
            backup_dir = v3_path / f"backup_unknowns_{split_name}_v3"
            create_directory(backup_dir)

            success_count = 0
            for img_file in tqdm(images_to_remove, desc=f"  Backup {split_name}"):
                try:
                    backup_path = backup_dir / img_file.name
                    shutil.move(str(img_file), str(backup_path))
                    success_count += 1
                except Exception as e:
                    print(f"    Erreur backup {img_file.name}: {e}")

            print(f"  ✓ {success_count} images unknowns déplacées vers backup")

        else:
            # Pas assez d'unknowns - ajouter depuis ImageNet
            missing = target_count - current_unknowns
            print(f"  📥 Ajout de {missing} images unknowns depuis ImageNet")
            
            success_count = add_imagenet_unknowns(unknowns_path, split_name, missing, current_unknowns)
            print(f"  ✓ {success_count} images ImageNet ajoutées comme unknowns")


def add_imagenet_unknowns(unknowns_path: Path, split_name: str, count_needed: int, start_counter: int) -> int:
    """Ajoute des images d'ImageNet comme unknowns."""
    imagenet_path = Path("/mnt/data/WORK/DATA/imagenet")
    
    if not imagenet_path.exists():
        print(f"    ❌ ImageNet non trouvé à {imagenet_path}")
        return 0
    
    # Chercher des images dans ImageNet/train
    imagenet_train = imagenet_path / "train"
    if not imagenet_train.exists():
        print(f"    ❌ Dossier ImageNet/train non trouvé")
        return 0
    
    # Collecter des images de différentes classes ImageNet
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG',
                       '.bmp', '.BMP', '.tiff', '.TIFF', '.webp', '.WEBP'}
    
    collected_images = []
    
    # Parcourir les classes ImageNet (n01440764, etc.)
    class_dirs = [d for d in imagenet_train.iterdir() if d.is_dir()]
    random.shuffle(class_dirs)  # Mélanger pour avoir de la diversité
    
    for class_dir in class_dirs:
        if len(collected_images) >= count_needed:
            break
            
        # Prendre quelques images de cette classe
        class_images = [f for f in class_dir.iterdir() 
                       if f.is_file() and f.suffix in image_extensions]
        
        if class_images:
            # Prendre max 3 images par classe pour diversifier
            sample_size = min(3, len(class_images), count_needed - len(collected_images))
            sampled = random.sample(class_images, sample_size)
            collected_images.extend(sampled)
    
    print(f"    📸 Trouvé {len(collected_images)} images ImageNet à ajouter")
    
    # Copier et convertir les images sélectionnées
    success_count = 0
    counter = start_counter + 1
    
    for img_file in tqdm(collected_images[:count_needed], desc=f"    Ajout ImageNet {split_name}"):
        try:
            new_filename = normalize_filename(
                img_file.name, "unknowns", split_name, counter, "imagenet"
            )
            target_path = unknowns_path / new_filename
            
            if convert_and_copy_image(img_file, target_path):
                success_count += 1
                counter += 1
        except Exception as e:
            print(f"      Erreur ajout {img_file.name}: {e}")
    
    return success_count


def process_synthetic_dataset(synthetic_path: Path, v3_path: Path):
    """Traite et intègre le dataset synthétique dans v3."""
    print(f"\n🤖 Intégration du dataset synthétique")

    # Explorer la structure
    structure = explore_synthetic_dataset(synthetic_path)

    if structure["total_images"] == 0:
        print("❌ Aucune image trouvée dans le dataset synthétique")
        return

    class_mapping = get_synthetic_class_mapping()

    # Compter les images existantes pour continuer la numérotation
    existing_counts = count_images_by_class(v3_path)

    stats = defaultdict(lambda: defaultdict(int))

    # Traiter chaque split trouvé
    for split_name in structure["splits"]:
        split_path = synthetic_path / split_name
        target_split = "valid" if split_name == "validation" else split_name
        target_split_path = v3_path / target_split

        if not split_path.exists():
            continue

        print(f"\n--- Traitement du split synthétique {split_name} -> {target_split} ---")

        # Explorer les classes dans ce split
        for item in split_path.iterdir():
            if not item.is_dir():
                continue

            # Identifier la classe de maturité
            target_class = None
            for synthetic_class, mapped_class in class_mapping.items():
                if synthetic_class in item.name or item.name == synthetic_class:
                    target_class = mapped_class
                    break

            if not target_class:
                print(f"  ⚠️  Classe non reconnue: {item.name}")
                continue

            # Créer le dossier de destination
            target_class_path = target_split_path / target_class
            create_directory(target_class_path)

            # Compter les images existantes pour continuer la numérotation
            counter = existing_counts.get(target_split, {}).get(target_class, 0) + 1

            # Traiter les images (y compris dans les sous-dossiers comme "nombre de bananes")
            image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG',
                               '.bmp', '.BMP', '.tiff', '.TIFF', '.webp', '.WEBP'}
            if HEIF_SUPPORT:
                image_extensions.update({'.heic', '.HEIC', '.heif', '.HEIF'})

            # Récupérer toutes les images (récursivement pour gérer les sous-dossiers)
            all_images = []
            for img_file in item.rglob("*"):
                if img_file.is_file() and img_file.suffix in image_extensions:
                    all_images.append(img_file)

            print(f"  {item.name} -> {target_class}: {len(all_images)} images")

            if not all_images:
                continue

            # Copier et convertir les images
            success_count = 0
            for img_file in tqdm(all_images, desc=f"  Conversion {item.name}"):
                new_filename = normalize_filename(
                    img_file.name, target_class, target_split, counter, "synthetic"
                )
                target_path = target_class_path / new_filename

                if convert_and_copy_image(img_file, target_path):
                    stats[target_split][target_class] += 1
                    success_count += 1
                    counter += 1

            # Mettre à jour les compteurs
            if target_split not in existing_counts:
                existing_counts[target_split] = {}
            existing_counts[target_split][target_class] = counter - 1

            print(f"    ✓ {success_count}/{len(all_images)} images synthétiques ajoutées")

    # Afficher le résumé
    print(f"\n📊 Résumé intégration synthétique:")
    for split_name, classes in stats.items():
        print(f"  {split_name.upper()}:")
        for class_name, count in classes.items():
            print(f"    - {class_name}: +{count} images synthétiques")


def print_final_stats_v3(v3_path: Path):
    """Affiche les statistiques finales du dataset v3."""
    print(f"\n=== Statistiques finales du dataset v3 ===")

    counts = count_images_by_class(v3_path)
    total_images = 0

    for split_name in ["train", "valid", "test"]:
        if split_name in counts:
            print(f"\n{split_name.upper()}:")
            split_total = 0

            for class_name in ["unripe", "ripe", "overripe", "rotten", "unknowns"]:
                if class_name in counts[split_name]:
                    count = counts[split_name][class_name]
                    print(f"  - {class_name}: {count} images")
                    split_total += count

            print(f"  Total {split_name}: {split_total} images")
            total_images += split_total

    print(f"\n🎉 TOTAL DATASET V3: {total_images} images")
    return total_images


def create_readme_v3(v3_path: Path):
    """Crée un README pour le dataset v3."""
    readme_content = """# Dataset v3 - Banana Ripeness Classification (avec données synthétiques)

## Sources des données

### Dataset v2 (base)
Dataset v2 utilisé comme base, incluant :
- Dataset v1 original
- Dataset gitbanana (https://github.com/luischuquim/BananaRipeness/)
- Dataset YOLO pour détection de bananes

### Dataset synthétique
Source: /mnt/data/WORK/DATA/Synthetic Dataset/

**Mapping des classes synthétiques:**
- Ripeness A → unripe (bananes vertes/pas mûres)
- Ripeness B → ripe (bananes mûres)  
- Ripeness C → overripe (bananes trop mûres)
- Ripeness D → rotten (bananes pourries)

**Nouveautés v3:**
- Intégration des images synthétiques
- Gestion du dossier "nombre de bananes"
- Équilibrage automatique de la classe unknowns

## Structure du dataset v3

```
datasetv3/
├── train/
│   ├── unripe/
│   ├── ripe/
│   ├── overripe/
│   ├── rotten/
│   └── unknowns/
├── valid/
│   ├── unripe/
│   ├── ripe/
│   ├── overripe/
│   ├── rotten/
│   └── unknowns/
└── test/
    ├── unripe/
    ├── ripe/
    ├── overripe/
    ├── rotten/
    └── unknowns/
```

## Format optimisé pour l'IA

- **Format d'image**: JPG uniforme (qualité 95)
- **Noms de fichiers**: Format uniformisé `{classe}_{split}_{numéro:06d}_source.jpg`
- **Sources identifiées**: `_synthetic` pour les images synthétiques
- **Orientations**: Corrigées automatiquement via EXIF
- **Mode RGB**: Toutes les images converties en RGB

Exemples: 
- `ripe_train_001234_synthetic.jpg` (image synthétique)
- `unripe_valid_000567.jpg` (image réelle)

## Équilibrage des classes

Le dataset v3 équilibre automatiquement la classe `unknowns` pour qu'elle soit 
proportionnelle aux autres classes dans chaque split.

## Date de création
Généré le: {date}

## Scripts utilisés
- create_datasetv3.py (avec intégration synthétique)

## Dépendances
- PIL/Pillow pour conversion d'images
- pillow-heif pour support HEIC/HEIF (optionnel)
"""

    from datetime import datetime
    readme_content = readme_content.format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    readme_path = v3_path / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"✓ README v3 créé: {readme_path}")


def main():
    # Chemins
    v2_path = Path("/mnt/data/WORK/DATA/datasetv2")
    synthetic_path = Path("/mnt/data/WORK/DATA/Synthetic Dataset")
    v3_path = Path("/mnt/data/WORK/DATA/datasetv3")

    print("🚀 CRÉATION DU DATASET V3")
    print("=" * 50)
    print(f"Dataset v2 (source): {v2_path}")
    print(f"Dataset synthétique: {synthetic_path}")
    print(f"Dataset v3 (destination): {v3_path}")

    # Fixer la seed pour la reproductibilité
    random.seed(42)
    #
    # # Étape 1: Copier v2 vers v3
    # if not copy_v2_to_v3(v2_path, v3_path):
    #     return
    #
    # # Étape 2: Explorer et intégrer le dataset synthétique
    # if synthetic_path.exists():
    #     process_synthetic_dataset(synthetic_path, v3_path)
    # else:
    #     print(f"⚠️  Dataset synthétique non trouvé à {synthetic_path}")
    #     print("Création de v3 sans données synthétiques")
    #
    # # Étape 3: Équilibrer les classes unknowns
    # balance_unknowns_class(v3_path)
    #
    # # Étape 4: Créer le README
    # create_readme_v3(v3_path)

    # Étape 5: Statistiques finales
    total_v3 = print_final_stats_v3(v3_path)

    print(f"\n✅ Dataset v3 créé avec succès !")
    print(f"📁 Disponible à: {v3_path}")
    print(f"📊 Total: {total_v3} images")


if __name__ == "__main__":
    main()
