from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Dict
import logging
import uuid

from ..services.predictor import get_predictor
from ..services.rate_limiter import limiter, PREDICT_LIMIT
from ..services.file_validator import is_avif_file, process_image_to_jpg

router = APIRouter(prefix="/predict", tags=["predict"])


class PredictFromS3Request(BaseModel):
    key: str


@router.post("/")
@limiter.limit(PREDICT_LIMIT)
async def predict_image(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    predictor = get_predictor()

    if is_avif_file(file.content_type or "", file.filename or ""):
        raise HTTPException(status_code=400, detail="Les fichiers AVIF ne sont pas acceptés")

    try:
        content = await file.read()
        processed_content = process_image_to_jpg(content)
        result = predictor.predict_image_bytes(processed_content)

        # Store temp file data for corrections
        import base64
        temp_file_data = {
            "content": base64.b64encode(processed_content).decode('utf-8'),
            "filename": f"{file.filename.split('.')[0]}.jpg" if file.filename else "image.jpg",
            "content_type": "image/jpeg"
        }

        result.update({
            "success": True,
            "filename": temp_file_data["filename"],
            "content_type": "image/jpeg",
            "temp_file_data": temp_file_data
        })
        result["image_key"] = f"temp_{uuid.uuid4().hex}"

        return result

    except Exception as e:
        logging.getLogger(__name__).error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors de la prédiction")


@router.post("/s3")
@limiter.limit(PREDICT_LIMIT)
async def predict_from_s3(request: Request, body: PredictFromS3Request) -> Dict[str, Any]:
    predictor = get_predictor()

    try:
        result = predictor.predict_from_s3(body.key)
        result["success"] = True
        return result

    except Exception as e:
        logging.getLogger(__name__).error(f"S3 prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors de la prédiction depuis S3")
