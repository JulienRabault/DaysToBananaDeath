#!/usr/bin/env python3
"""
Script pour intégrer le dataset gitbanana dans datasetv1 et créer datasetv2
avec des noms de fichiers uniformisés et format d'image optimisé pour l'IA.

Sources :
- Dataset v1 existant : /mnt/data/WORK/DATA/datasetv1/
- Dataset gitbanana : /mnt/data/WORK/DATA/gitbanana/
- Nouveau dataset : /mnt/data/WORK/DATA/datasetv2/

Mapping des classes gitbanana :
- Class A -> unripe
- Class B -> ripe  
- Class C -> overripe
- Class D -> rotten

Format optimisé pour l'IA :
- Toutes les images converties en JPG (qualité 95)
- Noms uniformisés: {classe}_{split}_{numéro:06d}.jpg
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List
import re
from tqdm import tqdm
from collections import defaultdict
from PIL import Image, ImageOps
import pillow_heif

# Enregistrer le support HEIF/HEIC (pour les iPhone)
pillow_heif.register_heif_opener()


def create_directory(path: Path):
    """Crée un dossier s'il n'existe pas."""
    path.mkdir(parents=True, exist_ok=True)


def convert_and_copy_image(src_path: Path, dest_path: Path, quality: int = 95) -> bool:
    """
    Convertit une image vers JPG avec qualité optimisée pour l'IA.

    Args:
        src_path: Chemin source de l'image
        dest_path: Chemin destination (sera forcé en .jpg)
        quality: Qualité JPG (95 recommandé pour l'IA)

    Returns:
        True si succès, False sinon
    """
    try:
        # Forcer l'extension destination en .jpg
        dest_path = dest_path.with_suffix('.jpg')

        # Ouvrir l'image
        with Image.open(src_path) as img:
            # Convertir en RGB si nécessaire (pour PNG avec transparence, etc.)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Créer un fond blanc pour les images avec transparence
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Corriger l'orientation EXIF si présente
            img = ImageOps.exif_transpose(img)

            # Sauvegarder en JPG avec qualité optimisée
            img.save(dest_path, 'JPEG', quality=quality, optimize=True)

        return True

    except Exception as e:
        print(f"    ❌ Erreur conversion {src_path.name}: {e}")
        return False


def copy_dataset_v1_to_v2(v1_path: Path, v2_path: Path):
    """Copie le dataset v1 vers v2 comme base."""
    print(f"=== Copie du dataset v1 vers v2 ===")
    
    if not v1_path.exists():
        print(f"Dataset v1 non trouvé à {v1_path}")
        return False
    
    if v2_path.exists():
        print(f"Dataset v2 existe déjà à {v2_path}")
        response = input("Voulez-vous le supprimer et recommencer ? (y/N): ")
        if response.lower() == 'y':
            shutil.rmtree(v2_path)
        else:
            return False
    
    # Copier tout le contenu de v1 vers v2
    shutil.copytree(v1_path, v2_path)
    print(f"✓ Dataset v1 copié vers v2")
    return True


def get_class_mapping() -> Dict[str, str]:
    """Retourne le mapping des classes gitbanana vers nos classes."""
    return {
        "Class A": "unripe",
        "Class B": "ripe", 
        "Class C": "overripe",
        "Class D": "rotten"
    }


def get_split_mapping() -> Dict[str, str]:
    """Mapping des splits gitbanana vers notre structure."""
    return {
        "train": "train",
        "validation": "valid", 
        "test": "test"
    }


def normalize_filename(filename: str, class_name: str, split_name: str, counter: int) -> str:
    """
    Génère un nom de fichier normalisé avec extension JPG.
    Format: {class}_{split}_{counter:06d}.jpg
    """
    # Toujours utiliser .jpg pour l'uniformité
    return f"{class_name}_{split_name}_{counter:06d}.jpg"


def count_existing_images(v2_path: Path) -> Dict[str, Dict[str, int]]:
    """Compte les images existantes dans v2 pour continuer la numérotation."""
    counts = defaultdict(lambda: defaultdict(int))
    
    for split_dir in v2_path.iterdir():
        if split_dir.is_dir() and split_dir.name in ["train", "valid", "test"]:
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    # Compter les fichiers existants
                    existing_files = list(class_dir.glob("*"))
                    counts[split_dir.name][class_dir.name] = len(existing_files)
    
    return dict(counts)


def process_gitbanana_split(gitbanana_split_path: Path, v2_base_path: Path, 
                           split_name: str, existing_counts: Dict[str, int]):
    """Traite un split du dataset gitbanana et l'ajoute à v2 avec conversion d'images."""

    if not gitbanana_split_path.exists():
        print(f"Split {split_name} non trouvé : {gitbanana_split_path}")
        return
    
    print(f"\n--- Traitement du split {split_name} ---")
    
    class_mapping = get_class_mapping()
    stats = defaultdict(int)
    conversion_stats = defaultdict(int)

    # Parcourir chaque classe
    for class_dir in gitbanana_split_path.iterdir():
        if class_dir.is_dir() and class_dir.name in class_mapping:
            target_class = class_mapping[class_dir.name]
            target_dir = v2_base_path / split_name / target_class
            create_directory(target_dir)
            
            # Obtenir le compteur de départ pour cette classe
            counter = existing_counts.get(target_class, 0) + 1
            
            # Extensions d'images supportées (élargi pour plus de formats)
            image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG',
                              '.bmp', '.BMP', '.tiff', '.TIFF', '.webp', '.WEBP',
                              '.heic', '.HEIC', '.heif', '.HEIF'}

            # Lister toutes les images dans le dossier de classe
            images = [f for f in class_dir.iterdir() 
                     if f.is_file() and f.suffix in image_extensions]
            
            print(f"  {class_dir.name} -> {target_class}: {len(images)} images")
            
            # Copier et convertir chaque image
            success_count = 0
            for img_file in tqdm(images, desc=f"  Conversion {class_dir.name}"):
                new_filename = normalize_filename(
                    img_file.name, target_class, split_name, counter
                )
                target_path = target_dir / new_filename
                
                # Convertir et copier l'image
                if convert_and_copy_image(img_file, target_path):
                    stats[target_class] += 1
                    success_count += 1
                    counter += 1

                    # Statistiques de conversion par format
                    original_ext = img_file.suffix.lower()
                    conversion_stats[original_ext] += 1

            # Mettre à jour le compteur pour la prochaine classe
            existing_counts[target_class] = counter - 1

            print(f"    ✓ {success_count}/{len(images)} images converties avec succès")

    # Afficher les statistiques
    print(f"Statistiques pour {split_name}:")
    for class_name, count in stats.items():
        print(f"  - {class_name}: +{count} images")

    if conversion_stats:
        print(f"Formats convertis:")
        for ext, count in conversion_stats.items():
            print(f"  - {ext} → .jpg: {count} images")


def create_readme(v2_path: Path):
    """Crée un README expliquant les sources du dataset."""
    readme_content = """# Dataset v2 - Banana Ripeness Classification

## Sources des données

### Dataset v1 (base)
Dataset original utilisé comme base pour ce dataset v2.

### Dataset gitbanana 
Source: https://github.com/luischuquim/BananaRipeness/

**Mapping des classes:**
- Class A → unripe (bananes vertes/pas mûres)
- Class B → ripe (bananes mûres)  
- Class C → overripe (bananes trop mûres)
- Class D → rotten (bananes pourries)

**Splits:**
- train → train/
- validation → valid/
- test → test/

## Structure du dataset v2

```
datasetv2/
├── train/
│   ├── unripe/
│   ├── ripe/
│   ├── overripe/
│   └── rotten/
├── valid/
│   ├── unripe/
│   ├── ripe/
│   ├── overripe/
│   └── rotten/
└── test/
    ├── unripe/
    ├── ripe/
    ├── overripe/
    └── rotten/
```

## Format optimisé pour l'IA

- **Format d'image**: JPG uniforme (qualité 95)
- **Noms de fichiers**: Format uniformisé `{classe}_{split}_{numéro:06d}.jpg`
- **Orientations**: Corrigées automatiquement via EXIF
- **Transparence**: Convertie sur fond blanc
- **Supports**: JPG, PNG, BMP, TIFF, WebP, HEIC/HEIF

Exemple: `ripe_train_001234.jpg`

## Optimisations IA

1. **Qualité JPG 95%**: Meilleur compromis taille/qualité pour l'entraînement
2. **Format uniforme**: Évite les problèmes de compatibilité entre formats
3. **Orientation corrigée**: Utilise les métadonnées EXIF pour l'orientation
4. **Mode RGB**: Toutes les images converties en RGB pour cohérence

## Date de création
Généré le: {date}

## Scripts utilisés
- integrate_gitbanana_dataset.py (avec conversion d'images)

## Dépendances
- PIL/Pillow pour conversion d'images
- pillow-heif pour support HEIC/HEIF
"""
    
    from datetime import datetime
    readme_content = readme_content.format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    readme_path = v2_path / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✓ README créé: {readme_path}")


def uniformize_filenames_v2(v2_path: Path):
    """
    Uniformise tous les noms de fichiers dans v2 et convertit vers JPG.
    """
    print(f"\n=== Uniformisation et conversion vers JPG ===")

    for split_dir in v2_path.iterdir():
        if split_dir.is_dir() and split_dir.name in ["train", "valid", "test"]:
            print(f"\n--- Split: {split_dir.name} ---")
            
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    
                    # Lister tous les fichiers images
                    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG',
                                      '.bmp', '.BMP', '.tiff', '.TIFF', '.webp', '.WEBP',
                                      '.heic', '.HEIC', '.heif', '.HEIF'}
                    images = [f for f in class_dir.iterdir()
                             if f.is_file() and f.suffix in image_extensions]
                    
                    print(f"  {class_name}: {len(images)} images à traiter")

                    # Créer un dossier temporaire pour éviter les conflits
                    temp_dir = class_dir.parent / f"{class_name}_temp"
                    create_directory(temp_dir)
                    
                    # Renommer et convertir tous les fichiers
                    counter = 1
                    success_count = 0
                    conversion_stats = defaultdict(int)

                    for img_file in tqdm(images, desc=f"    Traitement {class_name}"):
                        new_filename = normalize_filename(
                            img_file.name, class_name, split_dir.name, counter
                        )
                        temp_path = temp_dir / new_filename
                        
                        # Si c'est déjà un JPG avec le bon nom, juste le déplacer
                        if (img_file.suffix.lower() in ['.jpg', '.jpeg'] and
                            img_file.name == new_filename):
                            try:
                                shutil.move(str(img_file), str(temp_path))
                                success_count += 1
                                counter += 1
                            except Exception as e:
                                print(f"      Erreur déplacement {img_file.name}: {e}")
                        else:
                            # Convertir vers JPG
                            if convert_and_copy_image(img_file, temp_path):
                                success_count += 1
                                original_ext = img_file.suffix.lower()
                                conversion_stats[original_ext] += 1
                                counter += 1

                                # Supprimer l'original après conversion réussie
                                try:
                                    img_file.unlink()
                                except:
                                    pass

                    # Supprimer l'ancien dossier et renommer le temporaire
                    if class_dir.exists():
                        shutil.rmtree(class_dir)
                    temp_dir.rename(class_dir)
                    
                    print(f"    ✓ {success_count} fichiers traités")
                    if conversion_stats:
                        print(f"    Conversions:")
                        for ext, count in conversion_stats.items():
                            print(f"      {ext} → .jpg: {count}")


def print_final_stats(v2_path: Path):
    """Affiche les statistiques finales du dataset v2."""
    print(f"\n=== Statistiques finales du dataset v2 ===")
    
    total_images = 0
    
    for split_dir in ["train", "valid", "test"]:
        split_path = v2_path / split_dir
        if split_path.exists():
            print(f"\n{split_dir.upper()}:")
            split_total = 0
            
            for class_dir in ["unripe", "ripe", "overripe", "rotten", "unknowns"]:
                class_path = split_path / class_dir
                if class_path.exists():
                    count = len(list(class_path.glob("*")))
                    print(f"  - {class_dir}: {count} images")
                    split_total += count
            
            print(f"  Total {split_dir}: {split_total} images")
            total_images += split_total
    
    print(f"\n🎉 TOTAL DATASET V2: {total_images} images")


def main():
    # Chemins
    v1_path = Path("/mnt/data/WORK/DATA/datasetv1")
    gitbanana_path = Path("/mnt/data/WORK/DATA/gitbanana") 
    v2_path = Path("/mnt/data/WORK/DATA/datasetv2")
    
    print("=== Intégration du dataset gitbanana ===")
    print(f"Dataset v1: {v1_path}")
    print(f"Dataset gitbanana: {gitbanana_path}")
    print(f"Dataset v2 (sortie): {v2_path}")

    # Vérifier que gitbanana existe
    if not gitbanana_path.exists():
        print(f"❌ Dataset gitbanana non trouvé: {gitbanana_path}")
        return

    # Étape 1: Copier v1 vers v2 (ou créer v2 vide si v1 n'existe pas)
    if v1_path.exists():
        if not copy_dataset_v1_to_v2(v1_path, v2_path):
            return
    else:
        print("Dataset v1 non trouvé, création d'un dataset v2 vide")
        create_directory(v2_path)

        # Créer la structure de base
        for split in ["train", "valid", "test"]:
            for class_name in ["unripe", "ripe", "overripe", "rotten"]:
                create_directory(v2_path / split / class_name)

    # Étape 2: Compter les images existantes pour continuer la numérotation
    existing_counts = count_existing_images(v2_path)
    print(f"Images existantes comptées: {dict(existing_counts)}")

    # Étape 3: Intégrer gitbanana
    split_mapping = get_split_mapping()

    for gitbanana_split, target_split in split_mapping.items():
        gitbanana_split_path = gitbanana_path / gitbanana_split
        process_gitbanana_split(
            gitbanana_split_path, v2_path, target_split,
            existing_counts.get(target_split, {})
        )

    # Étape 4: Uniformiser tous les noms de fichiers
    uniformize_filenames_v2(v2_path)

    # Étape 6: Statistiques finales
    print_final_stats(v2_path)
    
    print(f"\n✅ Intégration terminée ! Dataset v2 disponible à: {v2_path}")


if __name__ == "__main__":
    main()
