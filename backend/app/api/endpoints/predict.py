from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

from ..services.predictor import get_predictor

router = APIRouter(prefix="/predict", tags=["predict"])


class PredictFromS3Request(BaseModel):
    key: str


@router.post("/file")
async def predict_from_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        content = await file.read()
        predictor = get_predictor()
        result = predictor.predict_image_bytes(content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @router.post("/s3")
# async def predict_from_s3(body: PredictFromS3Request) -> Dict[str, Any]:
#     try:
#         predictor = get_predictor()
#         result = predictor.predict_from_s3_key(body.key)
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
