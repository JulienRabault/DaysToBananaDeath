import os
import json
import tempfile
from datetime import datetime, timezone
from typing import Optional, Dict, Any

try:
    import boto3  # type: ignore
    from botocore.client import Config  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    boto3 = None  # type: ignore
    Config = None  # type: ignore

from ...config import config


class S3Service:
    def __init__(self):
        if boto3 is None:
            raise RuntimeError("boto3 is required for S3 operations. Install it via backend/requirements.txt")

        # Use config instead of os.getenv
        self.bucket = config.S3_BUCKET_NAME or os.getenv("S3_BUCKET")  # Backward compatibility
        region = config.AWS_REGION
        endpoint_url = config.S3_ENDPOINT_URL

        if not self.bucket:
            raise RuntimeError("S3_BUCKET_NAME or S3_BUCKET env var is required")

        session = boto3.session.Session(
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=region,
        )
        self.s3 = session.client("s3", endpoint_url=endpoint_url, config=Config(signature_version="s3v4"))

    def generate_presigned_post(self, key: str, expires_in: int = 3600, content_type: Optional[str] = None) -> Dict[str, Any]:
        conditions = []
        if content_type:
            conditions.append(["starts-with", "$Content-Type", content_type.split("/")[0]])
        resp = self.s3.generate_presigned_post(
            Bucket=self.bucket,
            Key=key,
            Fields={"Content-Type": content_type} if content_type else None,
            Conditions=conditions if conditions else None,
            ExpiresIn=expires_in,
        )
        return resp

    def generate_presigned_put(self, key: str, expires_in: int = 3600, content_type: Optional[str] = None) -> str:
        params = {"Bucket": self.bucket, "Key": key}
        if content_type:
            params["ContentType"] = content_type
        url = self.s3.generate_presigned_url("put_object", Params=params, ExpiresIn=expires_in)
        return url

    def upload_fileobj(self, file_obj, key: str, content_type: Optional[str] = None) -> str:
        extra_args = {"ContentType": content_type} if content_type else None
        self.s3.upload_fileobj(file_obj, self.bucket, key, ExtraArgs=extra_args or {})
        return key

    def download_to_tempfile(self, key: str) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        self.s3.download_fileobj(self.bucket, key, tmp)
        tmp.flush()
        tmp.close()
        return tmp.name

    def copy_object(self, source_key: str, dest_key: str) -> None:
        copy_source = {"Bucket": self.bucket, "Key": source_key}
        self.s3.copy(copy_source, self.bucket, dest_key)

    def put_json(self, key: str, data: Dict[str, Any]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType="application/json")

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            data = obj["Body"].read()
            return json.loads(data)
        except Exception:
            # Return None if not found or any other non-critical issue
            return None

    def increment_counter(self, key: str) -> int:
        data = self.get_json(key) or {"count": 0, "updated_at": None}
        data["count"] = int(data.get("count", 0)) + 1
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.put_json(key, data)
        return int(data["count"])  # assure int

    def ensure_prefix(self, prefix: str) -> str:
        return prefix.rstrip("/") + "/"


# Lazy singleton
_s3_service: Optional[S3Service] = None

def get_s3() -> S3Service:
    global _s3_service
    if _s3_service is None:
        _s3_service = S3Service()
    return _s3_service
