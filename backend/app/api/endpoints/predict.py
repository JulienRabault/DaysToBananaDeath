from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import logging
import uuid
import tempfile
import os

from ..services.predictor import get_predictor
from ..services.s3 import get_s3
from ...config import config

router = APIRouter(prefix="/predict", tags=["predict"])


class PredictFromS3Request(BaseModel):
    key: str


@router.post("/file")
async def predict_from_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        content = await file.read()

        # Make prediction directly without uploading to S3
        predictor = get_predictor()
        result = predictor.predict_image_bytes(content)

        # Store file info for potential correction later
        # We'll encode the file content in base64 for temporary storage
        import base64
        encoded_content = base64.b64encode(content).decode('utf-8')

        # Add file information to response (needed for corrections)
        result["temp_file_data"] = {
            "content": encoded_content,
            "filename": file.filename,
            "content_type": file.content_type or "image/jpeg"
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
