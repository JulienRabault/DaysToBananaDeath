import io
import logging
import time
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import onnxruntime as ort
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)

# Configuration simple
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Variables globales
model_session = None
transform = None
classes = ["overripe", "ripe", "rotten", "unripe", "unknowns"]
temp_dir = None  # Pour garder une référence au répertoire temporaire

def download_model_from_wandb():
    """Télécharge le modèle ONNX depuis W&B dans un répertoire temporaire."""
    global temp_dir

    try:
        logger.info("Téléchargement du modèle depuis W&B...")

        wandb.login()

        # Créer un répertoire temporaire qui persiste pendant la vie de l'application
        temp_dir = tempfile.TemporaryDirectory(prefix="wandb_artifacts_")
        logger.info(f"Répertoire temporaire créé: {temp_dir.name}")

        api = wandb.Api()
        artifact = api.artifact('jrabault/banana-classification-unknown/onnx:v0', type='model')
        artifact_dir = artifact.download(root=temp_dir.name)

        print(f"Artifact téléchargé dans: {artifact_dir}")

        # Trouver dynamiquement le fichier ONNX
        onnx_files = list(Path(artifact_dir).glob("*.onnx"))
        # onnx_files = ["/mnt/data/WORK/DaysToBananaDeath/outputs/banana-ripeness-classifier-unknowns/v2025.10.02/resnet50_lr1e-03_bs32_ep15_1759398575_909AXV/artifacts/model.onnx"]
        if not onnx_files:
            logger.error(f"Aucun fichier ONNX trouvé dans {artifact_dir}")
            return None

        if len(onnx_files) > 1:
            logger.warning(f"Plusieurs fichiers ONNX trouvés: {[f.name for f in onnx_files]}")
            logger.info(f"Utilisation du premier: {onnx_files[0].name}")

        model_path = str(onnx_files[0])
        print(f"Fichier ONNX trouvé: {model_path}")

        wandb.finish()

        return model_path

    except Exception as e:
        logger.error(f"Erreur téléchargement W&B: {e}")
        # Nettoyer le répertoire temporaire en cas d'erreur
        if temp_dir:
            temp_dir.cleanup()
            temp_dir = None
        return None

def setup_preprocessing():
    """Configuration simple des transformations d'image."""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisation et nettoyage de l'application."""
    global model_session, transform, temp_dir

    # Startup
    try:
        logger.info("Démarrage de l'API...")

        # Configuration des transformations
        transform = setup_preprocessing()

        # Téléchargement et chargement du modèle
        model_path = download_model_from_wandb()

        if model_path:
            model_session = ort.InferenceSession(model_path)
            logger.info("Modèle ONNX chargé avec succès!")
        else:
            logger.error("Impossible de charger le modèle")

    except Exception as e:
        logger.error(f"Erreur au démarrage: {e}")

    yield

    # Shutdown - Nettoyage automatique du répertoire temporaire
    logger.info("Arrêt de l'API...")
    if temp_dir:
        logger.info("Nettoyage du répertoire temporaire...")
        temp_dir.cleanup()
        logger.info("Répertoire temporaire supprimé")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Préparation de l'image pour le modèle."""
    if transform is None:
        raise ValueError("Transformations non initialisées")

    # Conversion en array numpy
    image_array = np.array(image)

    # Application des transformations
    transformed = transform(image=image_array)
    image_tensor = transformed['image']

    # Ajout de la dimension batch
    image_batch = np.expand_dims(image_tensor, axis=0)

    return image_batch

app = FastAPI(
    title="Banana Classifier",
    description="API simple pour classifier les bananes",
    lifespan=lifespan
)

@app.get("/")
async def home():
    """Page d'accueil simple."""
    return {"message": "API Banana Classifier - Envoyez une photo de banane!"}

@app.get("/health")
async def health():
    """Vérification simple de l'état."""
    is_ready = model_session is not None
    return {
        "status": "OK" if is_ready else "ERROR",
        "model_loaded": is_ready
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Prédiction simple à partir d'une image."""

    # Vérifications de base
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Veuillez envoyer une image!")

    if model_session is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé!")

    try:
        start_time = time.time()

        # Lecture de l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Conversion en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Préparation de l'image
        image_batch = preprocess_image(image)

        # Prédiction avec ONNX
        input_name = model_session.get_inputs()[0].name
        outputs = model_session.run(None, {input_name: image_batch})
        predictions = outputs[0][0]  # Premier batch, première prédiction

        # Calcul des probabilités
        exp_preds = np.exp(predictions)
        probabilities = exp_preds / np.sum(exp_preds)

        # Classe prédite
        predicted_idx = np.argmax(probabilities)
        predicted_class = classes[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        # Temps de traitement
        processing_time = round((time.time() - start_time) * 1000, 2)

        # Réponse simple
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                classes[i]: float(probabilities[i])
                for i in range(len(classes))
            },
            "processing_time_ms": processing_time
        }

    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

import os


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
