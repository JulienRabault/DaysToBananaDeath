# Days to banana death

## Contexte
Objectif principal :
Créer une application web capable de prédire, à partir d’une photo de banane, combien de jours il reste avant qu’elle ne devienne trop mûre ou pourrie. Le projet servira de terrain d’entraînement pour améliorer tes compétences en AI engineering, MLops et développement web rapide.

Données :

Dataset : Roboflow Banana Ripeness Classification

- Taille : ~12k images

- Classes : unripe, ripe, overripe, rotten

- Pipeline prévu :

Préprocessing des images : redimensionnement, normalisation, éventuellement augmentation.

Modélisation ML :

- Classification multi-classes (unripe → rotten) avec CNN ou modèle pré-entraîné (ResNet, EfficientNet).

- Optionnel : régression pour estimer les jours restants avant pourrissement.

Déploiement MLops :

- Entraînement et suivi via PyTorch / Lightning / MLflow.

- Versionning des modèles, logs de métriques.

API et Web :

- FastAPI pour servir le modèle.

- Endpoint universel pour upload ou prise de photo.

- Optionnel : interface web simple pour visualiser le “Days to Death”.

Extensions possibles :

- Suivi continu et ré-entraînement automatique (MLops pipeline).