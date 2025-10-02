# DaysToBananaDeath - Banana Ripeness Classification

## Description

Projet de classification de la maturité des bananes utilisant des techniques d'intelligence artificielle. Le système permet de déterminer automatiquement le stade de maturité d'une banane à partir d'images.

## Classes de maturité

- **unripe** : Bananes vertes/pas mûres
- **ripe** : Bananes mûres 
- **overripe** : Bananes trop mûres
- **rotten** : Bananes pourries
- **unknowns** : Images non classifiées

## Dataset

Ce projet utilise plusieurs sources de données :

### Dataset principal
Source : [BananaRipeness](https://github.com/luischuquim/BananaRipeness/)

**Citation :**
```bibtex
@conference{visapp23,
  author={Luis Chuquimarca. and Boris Vintimilla. and Sergio Velastin.},
  title={Banana Ripeness Level Classification Using a Simple CNN Model Trained with Real and Synthetic Datasets},
  booktitle={Proceedings of the 18th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 5: VISAPP, (VISIGRAPP 2023)},
  year={2023},
  pages={536-543},
  publisher={SciTePress},
  organization={INSTICC},
  doi={10.5220/0011654600003417},
  isbn={978-989-758-634-7},
  issn={2184-4321},
}
```

### Datasets additionnels
- Dataset YOLO pour détection de bananes
- Dataset v1 (base existante)
- Dataset v2 (version consolidée et optimisée)

## Structure du projet

```
DaysToBananaDeath/
├── src/                    # Code source
│   ├── api.py             # API REST
│   ├── model.py           # Modèles d'IA
│   ├── dataset.py         # Gestion des datasets
│   ├── train.py           # Entraînement
│   └── inference.py       # Inférence
├── configs/               # Configuration
├── scripts/               # Scripts utilitaires
└── outputs/               # Résultats d'entraînement
```

## Dataset v2 (Actuel)

**Statistiques finales :**
- **TRAIN** : 16,761 images
  - unripe: 3,152 images
  - ripe: 4,490 images  
  - overripe: 2,684 images
  - rotten: 4,435 images
  - unknowns: 2,000 images

- **VALID** : 2,268 images
  - unripe: 539 images
  - ripe: 611 images
  - overripe: 341 images
  - rotten: 527 images
  - unknowns: 250 images

- **TEST** : 1,784 images
  - unripe: 521 images
  - ripe: 465 images
  - overripe: 225 images
  - rotten: 323 images
  - unknowns: 250 images

**Total : 20,813 images**

## API

L'API REST permet de classifier des images de bananes en temps réel.

### Endpoints
- `POST /predict` : Classification d'une image
- `GET /health` : État de santé du service

## Installation

```bash
# Installation des dépendances
pip install -r requirements.txt

# Lancement de l'API
python src/api.py
```

## Utilisation

```python
# Exemple d'utilisation de l'API
import requests

# Classification d'une image
with open('banana.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/predict', 
                           files={'file': f})
    
result = response.json()
print(f"Classe prédite: {result['class']}")
print(f"Confiance: {result['confidence']}")
```

## Modèles

- **ResNet50** : Modèle de base avec transfer learning
- **ViT-B/16** : Vision Transformer pour comparaison

## Technologies

- **Python 3.13**
- **PyTorch** / **PyTorch Lightning**
- **FastAPI** pour l'API REST
- **Hydra** pour la configuration
- **Weights & Biases** pour le tracking des expériences

## Auteurs

- Équipe DaysToBananaDeath

## License

[À définir]

---

*Projet généré le 2 octobre 2025*
