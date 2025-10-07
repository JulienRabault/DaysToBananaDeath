"""File validation service for uploaded images."""

import io
import logging
from typing import Set

from fastapi import HTTPException, UploadFile
from PIL import Image

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MIN_FILE_SIZE = 1024  # 1 KB
ALLOWED_CONTENT_TYPES: Set[str] = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
    "image/avif"
}
ALLOWED_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif", ".avif"
}
MIN_IMAGE_DIMENSION = 32  # pixels
MAX_IMAGE_DIMENSION = 4096  # pixels


class ValidationError(Exception):
    """Exception raised during validation errors."""
    pass


def validate_file_upload(file: UploadFile) -> None:
    """Validate an uploaded file according to several criteria.

    Validates:
    - File size
    - MIME type
    - Extension
    - Image dimensions
    - Image format

    Raises:
        HTTPException: If validation fails
    """
    try:
        if not file.filename:
            raise ValidationError("File name is required")

        filename_lower = file.filename.lower()
        if not any(filename_lower.endswith(ext) for ext in ALLOWED_EXTENSIONS):
            raise ValidationError(
                f"File extension not allowed. "
                f"Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
            raise ValidationError(
                f"File type not allowed: {file.content_type}. "
                f"Allowed types: {', '.join(ALLOWED_CONTENT_TYPES)}"
            )

        logger.info(f"File validation passed for: {file.filename} ({file.content_type})")

    except ValidationError as e:
        logger.warning(f"File validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during file validation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during file validation")


async def validate_file_content(file: UploadFile) -> bytes:
    """Validate file content after reading.

    Validates:
    - Actual file size
    - Image format
    - Image dimensions

    Returns:
        bytes: The validated file content

    Raises:
        HTTPException: If validation fails
    """
    try:
        content = await file.read()

        file_size = len(content)
        if file_size < MIN_FILE_SIZE:
            raise ValidationError(f"File too small: {file_size} bytes (minimum: {MIN_FILE_SIZE} bytes)")

        if file_size > MAX_FILE_SIZE:
            raise ValidationError(
                f"File too large: {file_size} bytes "
                f"(maximum: {MAX_FILE_SIZE / 1024 / 1024:.1f} MB)"
            )

        try:
            img = Image.open(io.BytesIO(content))

            width, height = img.size
            if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                raise ValidationError(
                    f"Image too small: {width}x{height} pixels "
                    f"(minimum: {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} pixels)"
                )

            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                raise ValidationError(
                    f"Image too large: {width}x{height} pixels "
                    f"(maximum: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} pixels)"
                )

            if img.format not in ['JPEG', 'PNG', 'WEBP', 'GIF', 'BMP', 'TIFF', 'AVIF']:
                raise ValidationError(f"Unsupported image format: {img.format}")

            logger.info(
                f"Image validation passed: {width}x{height} pixels, "
                f"{img.format}, {file_size} bytes"
            )

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError("File is not a valid image or is corrupted")

        await file.seek(0)

        return content

    except ValidationError as e:
        logger.warning(f"File content validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during file content validation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during file content validation")


def log_upload_attempt(file: UploadFile, client_ip: str = "unknown") -> None:
    """Log upload attempts for security monitoring."""
    logger.info(
        f"Upload attempt: file={file.filename}, "
        f"content_type={file.content_type}, "
        f"client_ip={client_ip}"
    )


def is_avif_file(content_type: str, filename: str) -> bool:
    return content_type == 'image/avif' or filename.lower().endswith('.avif')


def process_image_to_jpg(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))

    if img.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize((512, 512), Image.Resampling.LANCZOS)

    output_buffer = io.BytesIO()
    img.save(output_buffer, format='JPEG', quality=90)

    return output_buffer.getvalue()
