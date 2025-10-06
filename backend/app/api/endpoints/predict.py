from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Dict
import logging
import uuid
import base64
import io
from PIL import Image

from ..services.predictor import get_predictor
from ..services.rate_limiter import limiter, PREDICT_LIMIT
from ..services.file_validator import validate_file_upload, validate_file_content, log_upload_attempt

router = APIRouter(prefix="/predict", tags=["predict"])


def optimize_image_for_training(image_bytes: bytes, original_filename: str = "image") -> tuple[bytes, str]:
    """
    Convert any image to optimal format for training.
    Function duplicated from correction.py to avoid circular imports.
    """
    logger = logging.getLogger(__name__)

    try:
        img = Image.open(io.BytesIO(image_bytes))

        original_format = img.format
        original_size = img.size
        original_mode = img.mode

        logger.info(f"[PREDICT_OPT] Original: {original_format} {original_size} {original_mode}")

        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        max_size = 2048
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            logger.info(f"[PREDICT_OPT] Resized to: {img.size}")

        output_buffer = io.BytesIO()

        img.save(
            output_buffer,
            format='JPEG',
            quality=90,
            optimize=True,
            progressive=True
        )

        optimized_bytes = output_buffer.getvalue()

        base_name = os.path.splitext(original_filename)[0]
        new_filename = f"{base_name}_optimized.jpg"

        original_size_bytes = len(image_bytes)
        optimized_size_bytes = len(optimized_bytes)
        compression_ratio = (1 - optimized_size_bytes / original_size_bytes) * 100

        logger.info(f"[PREDICT_OPT] Optimized: JPEG {img.size} RGB")
        logger.info(f"[PREDICT_OPT] Size: {original_size_bytes} -> {optimized_size_bytes} bytes ({compression_ratio:.1f}% reduction)")
        logger.info(f"[PREDICT_OPT] Filename: {original_filename} -> {new_filename}")

        return optimized_bytes, new_filename

    except Exception as e:
        logger.error(f"[PREDICT_OPT] Failed to optimize image: {str(e)}")
        return image_bytes, original_filename


class PredictFromS3Request(BaseModel):
    """Request model for S3-based predictions."""
    key: str


@router.post("/file")
@limiter.limit(PREDICT_LIMIT)
async def predict_from_file(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    """Predict banana ripeness from uploaded file."""
    try:
        client_ip = request.client.host if request.client else "unknown"
        log_upload_attempt(file, client_ip)

        validate_file_upload(file)
        content = await validate_file_content(file)

        optimized_content, optimized_filename = optimize_image_for_training(
            content,
            file.filename or "uploaded_image"
        )

        predictor = get_predictor()
        result = predictor.predict_image_bytes(optimized_content)

        encoded_content = base64.b64encode(optimized_content).decode('utf-8')

        result["temp_file_data"] = {
            "content": encoded_content,
            "filename": optimized_filename,
            "content_type": "image/jpeg"
        }
        result["image_key"] = f"temp_{uuid.uuid4().hex}"

        return result

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in predict_from_file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
