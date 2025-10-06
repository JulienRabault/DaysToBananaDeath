from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
import uuid
import posixpath
from datetime import datetime, timezone
import json as _json
from urllib import request as _urlreq
import base64
import io
from PIL import Image

from ..services.s3 import get_s3
from ..services.predictor import get_predictor
from ..services.rate_limiter import limiter, CORRECTION_LIMIT
from ..services.recaptcha import verify_recaptcha_token, RecaptchaError, is_recaptcha_enabled
from ...config import config

router = APIRouter(prefix="/corrections", tags=["corrections"])

CLASS_NAMES = ['overripe', 'ripe', 'rotten', 'unripe', 'unknowns']


def optimize_image_for_training(image_bytes: bytes, original_filename: str = "image") -> tuple[bytes, str]:
    """
    Convert any image to optimal format for training.

    Args:
        image_bytes: Raw image data
        original_filename: Original filename to preserve some info

    Returns:
        tuple: (optimized_image_bytes, new_filename)
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        # Open the image with PIL
        img = Image.open(io.BytesIO(image_bytes))

        # Original image info
        original_format = img.format
        original_size = img.size
        original_mode = img.mode

        logger.info(f"[IMAGE_OPT] Original: {original_format} {original_size} {original_mode}")

        # Convert to RGB if necessary (to support RGBA, P, etc.)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create a white background for images with transparency
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize if the image is too large (training optimization)
        max_size = 512  # Recommended max size for training
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            logger.info(f"[IMAGE_OPT] Resized to: {img.size}")

        # Save as optimized JPEG
        output_buffer = io.BytesIO()

        # Optimal quality for training: 90%
        # - High enough to preserve important details
        # - Compressed enough to reduce storage size
        img.save(
            output_buffer,
            format='JPEG',
            quality=90,
            optimize=True,
            progressive=True  # Progressive JPEG for better performance
        )

        optimized_bytes = output_buffer.getvalue()

        # Generate new filename
        base_name = os.path.splitext(original_filename)[0]
        new_filename = f"{base_name}_optimized.jpg"

        # Compression statistics
        original_size_bytes = len(image_bytes)
        optimized_size_bytes = len(optimized_bytes)
        compression_ratio = (1 - optimized_size_bytes / original_size_bytes) * 100

        logger.info(f"[IMAGE_OPT] Optimized: JPEG {img.size} RGB")
        logger.info(f"[IMAGE_OPT] Size: {original_size_bytes} -> {optimized_size_bytes} bytes ({compression_ratio:.1f}% reduction)")
        logger.info(f"[IMAGE_OPT] Filename: {original_filename} -> {new_filename}")

        return optimized_bytes, new_filename

    except Exception as e:
        logger.error(f"[IMAGE_OPT] Failed to optimize image: {str(e)}")
        # In case of error, return the original image
        return image_bytes, original_filename


class CorrectionRequest(BaseModel):
    """Request model for image correction submissions."""
    image_key: str = Field(..., description="S3 key or temporary ID of the uploaded image")
    temp_file_data: Optional[Dict[str, str]] = Field(None, description="Temporary file data for images not yet uploaded to S3")
    recaptcha_token: Optional[str] = Field(None, description="reCAPTCHA v3 token for bot protection")
    corrected_label: Optional[str] = Field(None, description="Corrected label among supported classes")
    is_banana: Optional[bool] = Field(None, description="True if it's a banana, False otherwise")
    days_left: Optional[float] = Field(None, description="Number of days left if it's a banana")
    predicted_label: Optional[str] = None
    predicted_index: Optional[int] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@router.post("")
@limiter.limit(CORRECTION_LIMIT)
async def submit_correction(request: Request, body: CorrectionRequest) -> Dict[str, Any]:
    """Submit a correction for a predicted image."""
    import logging

    # Check if corrections are enabled
    if not config.ENABLE_CORRECTIONS:
        logging.warning("[CORRECTION] Corrections are disabled via ENABLE_CORRECTIONS=false")
        raise HTTPException(
            status_code=503,
            detail="Correction service is temporarily disabled"
        )

    # reCAPTCHA verification for corrections
    if is_recaptcha_enabled():
        if not body.recaptcha_token:
            raise HTTPException(
                status_code=400,
                detail="reCAPTCHA token required to submit correction"
            )

        try:
            recaptcha_result = await verify_recaptcha_token(body.recaptcha_token, "correction")
            logging.info(f"[CORRECTION] reCAPTCHA verified: score={recaptcha_result.get('score', 'N/A')}")
        except RecaptchaError as e:
            logging.warning(f"[CORRECTION] reCAPTCHA verification failed: {str(e)}")
            raise HTTPException(status_code=403, detail=f"Bot verification failed: {str(e)}")

    # Enhanced logging for debugging
    logging.info(f"[CORRECTION] === DEBUGGING CORRECTION REQUEST ===")
    logging.info(f"[CORRECTION] image_key: {body.image_key}")
    logging.info(f"[CORRECTION] temp_file_data present: {bool(body.temp_file_data)}")
    if body.temp_file_data:
        logging.info(f"[CORRECTION] temp_file_data keys: {list(body.temp_file_data.keys())}")
        logging.info(f"[CORRECTION] filename: {body.temp_file_data.get('filename', 'NO FILENAME')}")
        logging.info(f"[CORRECTION] content_type: {body.temp_file_data.get('content_type', 'NO CONTENT_TYPE')}")
        logging.info(f"[CORRECTION] content length: {len(body.temp_file_data.get('content', ''))}")
    else:
        logging.error(f"[CORRECTION] NO TEMP_FILE_DATA - This will cause 404 error!")

    # Validate image_key is not empty
    if not body.image_key or body.image_key.strip() == "":
        raise HTTPException(status_code=400, detail="image_key is required and cannot be empty")

    predictor = get_predictor()

    if body.corrected_label:
        if body.corrected_label not in CLASS_NAMES:
            raise HTTPException(status_code=400, detail=f"corrected_label must be one of {CLASS_NAMES}")
        final_label = body.corrected_label
    else:
        if body.is_banana is None:
            raise HTTPException(status_code=400, detail="Provide either corrected_label or is_banana (+ days_left if True)")
        if body.is_banana is False:
            final_label = 'unknowns'
        else:
            if body.days_left is None:
                raise HTTPException(status_code=400, detail="days_left is required when is_banana is True")

            final_label = predictor.get_class_from_days_left(float(body.days_left))

            # Optional: validate the correction
            if body.corrected_label:
                validation_info = predictor.validate_correction(float(body.days_left), body.corrected_label)

    s3 = get_s3()

    dataset_prefix = config.DATASET_PREFIX
    dataset_prefix = s3.ensure_prefix(dataset_prefix)
    corrections_prefix = config.CORRECTIONS_PREFIX
    corrections_prefix = s3.ensure_prefix(corrections_prefix)
    counter_key = config.CORRECTION_COUNTER_KEY
    threshold = config.CORRECTION_THRESHOLD

    # Generate a new ID and structured destination
    cid = uuid.uuid4().hex

    try:
        # Handle temporary images vs S3 images
        source_key = body.image_key

        # Log the incoming request for debugging
        import logging
        logging.info(f"[CORRECTION] Processing correction for image_key: {body.image_key}")
        logging.info(f"[CORRECTION] Has temp_file_data: {bool(body.temp_file_data)}")

        if body.image_key.startswith("temp_") and body.temp_file_data:
            # This is a temporary image, upload it to S3 first
            import base64
            import io

            logging.info("[CORRECTION] Processing temporary image")

            # Decode the base64 content
            image_content = base64.b64decode(body.temp_file_data["content"])
            filename = body.temp_file_data.get("filename", "image.jpg")
            content_type = body.temp_file_data.get("content_type", "image/jpeg")

            # Optimize image for training
            optimized_image, new_filename = optimize_image_for_training(image_content, filename)

            # Force JPG extension for all corrections
            ext = ".jpg"

            # Upload directly to dataset structure (no need for intermediate upload)
            dest_key = posixpath.join(dataset_prefix, final_label, f"{cid}{ext}")

            # Upload the image directly to final destination
            s3.upload_fileobj(
                io.BytesIO(optimized_image),
                dest_key,
                content_type="image/jpeg"  # Force JPEG content type
            )

            # For record keeping, use the destination as source
            source_key = dest_key

            logging.info(f"[CORRECTION] Temporary image uploaded directly to: {dest_key}")

        else:
            # This is an existing S3 image, copy it
            logging.info(f"[CORRECTION] Processing existing S3 image: {source_key}")

            # For existing S3 images, we need to download, optimize and re-upload as JPG
            try:
                # Download the original image
                image_data = s3.download_fileobj(source_key)

                # Get original filename for optimization
                original_filename = os.path.basename(source_key)

                # Optimize image for training (converts to JPG)
                optimized_image, new_filename = optimize_image_for_training(image_data, original_filename)

                # Force JPG extension for all corrections
                ext = ".jpg"

                # Destination in dataset structure
                dest_key = posixpath.join(dataset_prefix, final_label, f"{cid}{ext}")

                # Upload the optimized JPG image
                s3.upload_fileobj(
                    io.BytesIO(optimized_image),
                    dest_key,
                    content_type="image/jpeg"
                )

                logging.info(f"[CORRECTION] S3 image downloaded, optimized and uploaded as JPG to: {dest_key}")

            except Exception as e:
                logging.error(f"[CORRECTION] Failed to process existing S3 image: {str(e)}")
                # Fallback: simple copy with JPG extension
                ext = ".jpg"
                dest_key = posixpath.join(dataset_prefix, final_label, f"{cid}{ext}")
                s3.copy_object(source_key=source_key, dest_key=dest_key)
                logging.info(f"[CORRECTION] Fallback: S3 image copied from {source_key} to {dest_key}")

        # Record JSON
        record = {
            "id": cid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_key": source_key,
            "dest_key": dest_key,
            "final_label": final_label,
            "provided_corrected_label": body.corrected_label,
            "is_banana": body.is_banana,
            "days_left": body.days_left,
            "predicted_label": body.predicted_label,
            "predicted_index": body.predicted_index,
            "confidence": body.confidence,
            "metadata": body.metadata or {},
            "was_temp_upload": body.image_key.startswith("temp_")
        }
        record_key = posixpath.join(corrections_prefix, "records", f"{cid}.json")
        s3.put_json(record_key, record)

        # Increment counter
        count = s3.increment_counter(counter_key)
        threshold_reached = count >= threshold

        # Optional: alert webhook
        if threshold_reached:
            webhook = config.ALERT_WEBHOOK_URL
            if webhook:
                payload = {
                    "event": "corrections_threshold_reached",
                    "count": count,
                    "threshold": threshold,
                    "latest_record_key": record_key,
                    "latest_dest_key": dest_key,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    req = _urlreq.Request(webhook, data=_json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
                    _urlreq.urlopen(req, timeout=5)
                except Exception:
                    # Don't block the response if the alert fails
                    pass

        return {
            "ok": True,
            "record_key": record_key,
            "dest_key": dest_key,
            "source_key": source_key,
            "label": final_label,
            "count": count,
            "threshold": threshold,
            "threshold_reached": threshold_reached,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def corrections_stats() -> Dict[str, Any]:
    """Get correction statistics."""
    # Check if corrections are enabled
    if not config.ENABLE_CORRECTIONS:
        raise HTTPException(
            status_code=503,
            detail="Correction service is temporarily disabled"
        )

    s3 = get_s3()
    counter_key = config.CORRECTION_COUNTER_KEY
    data = s3.get_json(counter_key) or {"count": 0}
    return {"count": int(data.get("count", 0)), "updated_at": data.get("updated_at")}
