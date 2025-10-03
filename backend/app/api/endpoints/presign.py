from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import uuid
from datetime import datetime, timezone

from ..services.s3 import get_s3

router = APIRouter(prefix="/presign", tags=["upload"])


class PresignRequest(BaseModel):
    content_type: Optional[str] = None
    folder: Optional[str] = None  # e.g., "incoming"
    filename: Optional[str] = None
    use_put: bool = True  # future: support POST


@router.post("/upload")
async def presign_upload(body: PresignRequest) -> Dict[str, Any]:
    try:
        s3 = get_s3()
        folder = body.folder or os.getenv("UPLOAD_PREFIX", "incoming")
        folder = s3.ensure_prefix(folder)
        name = body.filename or f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}"

        # Infer extension from content-type if available
        ext = ""
        if body.content_type:
            mapping = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/webp": ".webp",
                "image/bmp": ".bmp",
            }
            ext = mapping.get(body.content_type.lower(), "")
        key = f"{folder}{name}{ext}"

        if body.use_put:
            url = s3.generate_presigned_put(key=key, content_type=body.content_type)
            return {"method": "PUT", "url": url, "key": key, "bucket": s3.bucket}
        else:
            post = s3.generate_presigned_post(key=key, content_type=body.content_type)
            return {"method": "POST", "post": post, "key": key, "bucket": s3.bucket}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
