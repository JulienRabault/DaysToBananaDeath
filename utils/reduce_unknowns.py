#!/usr/bin/env python3
"""
Script pour réduire le nombre d'images 'unknowns' dans les splits valid et test à 250 maximum.
"""

import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

def reduce_unknowns_in_split(split_path: Path, split_name: str, max_unknowns: int = 250):
    """
    Réduit le nombre d'images unknowns dans un split à un maximum donné.

    Args:
        split_path: Chemin vers le split (valid ou test)
        split_name: Nom du split pour l'affichage
        max_unknowns: Nombre maximum d'images unknowns à conserver
    """
    unknowns_path = split_path / "unknowns"

    if not unknowns_path.exists():
        print(f"❌ Dossier unknowns non trouvé dans {split_name}")
        return

    # Lister toutes les images dans unknowns
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG',
                       '.bmp', '.BMP', '.tiff', '.TIFF', '.webp', '.WEBP'}

    all_images = [f for f in unknowns_path.iterdir()
                  if f.is_file() and f.suffix in image_extensions]

    current_count = len(all_images)
    print(f"\n--- {split_name.upper()} ---")
    print(f"Images unknowns actuelles: {current_count}")

    if current_count <= max_unknowns:
        print(f"✓ Déjà dans la limite ({max_unknowns}), rien à faire")
        return

    # Calculer combien d'images à supprimer
    to_remove = current_count - max_unknowns
    print(f"Images à supprimer: {to_remove}")

    # Créer un dossier de sauvegarde pour les images supprimées (optionnel)
    backup_dir = split_path.parent / f"backup_unknowns_{split_name}"
    backup_dir.mkdir(exist_ok=True)

    # Mélanger la liste pour une sélection aléatoire
    random.shuffle(all_images)

    # Sélectionner les images à supprimer
    images_to_remove = all_images[:to_remove]
    images_to_keep = all_images[to_remove:]

    print(f"Déplacement de {len(images_to_remove)} images vers backup...")

    # Déplacer les images vers le dossier de backup
    success_count = 0
    for img_file in tqdm(images_to_remove, desc=f"Nettoyage {split_name}"):
        try:
            # Déplacer vers backup au lieu de supprimer
            backup_path = backup_dir / img_file.name
            shutil.move(str(img_file), str(backup_path))
            success_count += 1
        except Exception as e:
            print(f"Erreur déplacement {img_file.name}: {e}")

    # Vérifier le résultat
    remaining_images = [f for f in unknowns_path.iterdir()
                       if f.is_file() and f.suffix in image_extensions]
    final_count = len(remaining_images)

    print(f"✓ {success_count} images déplacées vers backup")
    print(f"✓ Images unknowns restantes: {final_count}")
    print(f"✓ Backup créé dans: {backup_dir}")

    return final_count

def print_current_stats(v2_path: Path):
    """Affiche les statistiques actuelles du dataset v2."""
    print("=== STATISTIQUES ACTUELLES ===")

    for split_name in ["train", "valid", "test"]:
        split_path = v2_path / split_name
        if split_path.exists():
            print(f"\n{split_name.upper()}:")
            split_total = 0

            for class_name in ["unripe", "ripe", "overripe", "rotten", "unknowns"]:
                class_path = split_path / class_name
                if class_path.exists():
                    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG',
                                      '.bmp', '.BMP', '.tiff', '.TIFF', '.webp', '.WEBP'}
                    images = [f for f in class_path.iterdir()
                             if f.is_file() and f.suffix in image_extensions]
                    count = len(images)

                    if count > 0:
                        print(f"  - {class_name}: {count} images")
                        split_total += count

            print(f"  Total {split_name}: {split_total} images")

def main():
    # Chemin vers le dataset v2
    v2_path = Path("/mnt/data/WORK/DATA/datasetv2")

    if not v2_path.exists():
        print(f"❌ Dataset v2 non trouvé à {v2_path}")
        return

    print("🧹 RÉDUCTION DES IMAGES UNKNOWNS")
    print("=" * 50)

    # Fixer la seed pour la reproductibilité
    random.seed(42)

    # Afficher les statistiques actuelles
    print_current_stats(v2_path)

    print(f"\n🎯 OBJECTIF: Réduire unknowns à max 250 dans valid et test")

    # Réduire unknowns dans valid
    valid_path = v2_path / "valid"
    if valid_path.exists():
        reduce_unknowns_in_split(valid_path, "valid", max_unknowns=250)

    # Réduire unknowns dans test
    test_path = v2_path / "test"
    if test_path.exists():
        reduce_unknowns_in_split(test_path, "test", max_unknowns=250)

    print(f"\n" + "=" * 50)
    print("📊 STATISTIQUES FINALES")
    print("=" * 50)

    # Afficher les statistiques finales
    print_current_stats(v2_path)

    print(f"\n✅ Nettoyage terminé !")
    print(f"📁 Les images supprimées sont sauvegardées dans les dossiers backup_unknowns_*")

if __name__ == "__main__":
    main()
