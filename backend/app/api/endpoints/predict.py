from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import logging
import uuid
import tempfile
import os
import base64
import io
from PIL import Image

from ..services.predictor import get_predictor
from ..services.s3 import get_s3
from ...config import config

router = APIRouter(prefix="/predict", tags=["predict"])


def optimize_image_for_training(image_bytes: bytes, original_filename: str = "image") -> tuple[bytes, str]:
    """
    Convertit n'importe quelle image vers le format optimal pour l'entraînement.
    Fonction dupliquée depuis correction.py pour éviter les imports circulaires.
    """
    logger = logging.getLogger(__name__)

    try:
        # Ouvrir l'image avec PIL
        img = Image.open(io.BytesIO(image_bytes))

        # Informations sur l'image originale
        original_format = img.format
        original_size = img.size
        original_mode = img.mode

        logger.info(f"[PREDICT_OPT] Original: {original_format} {original_size} {original_mode}")

        # Convertir en RGB si nécessaire (pour supporter RGBA, P, etc.)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Créer un fond blanc pour les images avec transparence
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Redimensionner si l'image est trop grande (optimisation pour l'entraînement)
        max_size = 2048  # Taille max recommandée pour l'entraînement
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            logger.info(f"[PREDICT_OPT] Resized to: {img.size}")

        # Sauvegarder en JPEG optimisé
        output_buffer = io.BytesIO()

        # Qualité optimale pour l'entraînement : 90%
        img.save(
            output_buffer,
            format='JPEG',
            quality=90,
            optimize=True,
            progressive=True
        )

        optimized_bytes = output_buffer.getvalue()

        # Générer nouveau nom de fichier
        base_name = os.path.splitext(original_filename)[0]
        new_filename = f"{base_name}_optimized.jpg"

        # Statistiques de compression
        original_size_bytes = len(image_bytes)
        optimized_size_bytes = len(optimized_bytes)
        compression_ratio = (1 - optimized_size_bytes / original_size_bytes) * 100

        logger.info(f"[PREDICT_OPT] Optimized: JPEG {img.size} RGB")
        logger.info(f"[PREDICT_OPT] Size: {original_size_bytes} -> {optimized_size_bytes} bytes ({compression_ratio:.1f}% reduction)")
        logger.info(f"[PREDICT_OPT] Filename: {original_filename} -> {new_filename}")

        return optimized_bytes, new_filename

    except Exception as e:
        logger.error(f"[PREDICT_OPT] Failed to optimize image: {str(e)}")
        # En cas d'erreur, retourner l'image originale
        return image_bytes, original_filename


class PredictFromS3Request(BaseModel):
    key: str


@router.post("/file")
async def predict_from_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        content = await file.read()

        # Optimize image for training consistency
        optimized_content, optimized_filename = optimize_image_for_training(
            content,
            file.filename or "uploaded_image"
        )

        # Make prediction with optimized image
        predictor = get_predictor()
        result = predictor.predict_image_bytes(optimized_content)

        # Store optimized file info for potential correction later
        encoded_content = base64.b64encode(optimized_content).decode('utf-8')

        # Add file information to response (needed for corrections)
        result["temp_file_data"] = {
            "content": encoded_content,
            "filename": optimized_filename,
            "content_type": "image/jpeg"  # Always JPEG after optimization
        }
        # For compatibility with frontend, we'll use a temporary identifier
        result["image_key"] = f"temp_{uuid.uuid4().hex}"

        return result

    except Exception as e:
        logging.error(f"Erreur dans predict_from_file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


# @router.post("/s3")
# async def predict_from_s3(body: PredictFromS3Request) -> Dict[str, Any]:
#     try:
#         predictor = get_predictor()
#         result = predictor.predict_from_s3_key(body.key)
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
