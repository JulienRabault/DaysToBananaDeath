"""Service for S3 operations including uploads, downloads, and presigned URLs."""

import os
import json
import tempfile
from datetime import datetime, timezone
from typing import Optional, Dict, Any

try:
    import boto3
    from botocore.client import Config
except Exception:
    boto3 = None
    Config = None

from ...config import config


class S3Service:
    """Service for S3 operations including uploads, downloads, and presigned URLs."""

    def __init__(self) -> None:
        if boto3 is None:
            raise RuntimeError("boto3 is required for S3 operations. Install it via backend/requirements.txt")

        self.bucket = config.S3_BUCKET_NAME or os.getenv("S3_BUCKET")
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
        """Generate presigned POST URL for file uploads."""
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
        """Generate presigned PUT URL for file uploads."""
        params = {"Bucket": self.bucket, "Key": key}
        if content_type:
            params["ContentType"] = content_type
        return self.s3.generate_presigned_url("put_object", Params=params, ExpiresIn=expires_in)

    def upload_fileobj(self, fileobj: Any, key: str, content_type: Optional[str] = None) -> None:
        """Upload file object to S3."""
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        self.s3.upload_fileobj(fileobj, self.bucket, key, ExtraArgs=extra_args)

    def download_fileobj(self, key: str) -> bytes:
        """Download file from S3 as bytes."""
        import io
        buf = io.BytesIO()
        self.s3.download_fileobj(self.bucket, key, buf)
        return buf.getvalue()

    def download_to_tempfile(self, key: str) -> str:
        """Download file from S3 to temporary file and return path."""
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        self.s3.download_file(self.bucket, key, tmp.name)
        return tmp.name

    def copy_object(self, source_key: str, dest_key: str) -> None:
        """Copy object within S3 bucket."""
        copy_source = {"Bucket": self.bucket, "Key": source_key}
        self.s3.copy_object(CopySource=copy_source, Bucket=self.bucket, Key=dest_key)

    def put_json(self, key: str, data: Dict[str, Any]) -> None:
        """Upload JSON data to S3."""
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(data, default=str),
            ContentType="application/json"
        )

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Download and parse JSON from S3."""
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(response["Body"].read())
        except self.s3.exceptions.NoSuchKey:
            return None

    def object_exists(self, key: str) -> bool:
        """Check if an object exists in S3."""
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.s3.exceptions.NoSuchKey:
            return False
        except Exception:
            return False

    def increment_counter(self, key: str) -> int:
        """Atomically increment counter stored in S3."""
        try:
            data = self.get_json(key) or {"count": 0}
            count = int(data.get("count", 0)) + 1
            data["count"] = count
            data["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.put_json(key, data)
            return count
        except Exception:
            self.put_json(key, {"count": 1, "updated_at": datetime.now(timezone.utc).isoformat()})
            return 1

    def ensure_prefix(self, prefix: str) -> str:
        """Ensure prefix ends with '/' for S3 key structure."""
        return prefix if prefix.endswith("/") else f"{prefix}/"


_s3_singleton: Optional[S3Service] = None


def get_s3() -> S3Service:
    """Get the global S3 service instance."""
    global _s3_singleton
    if _s3_singleton is None:
        _s3_singleton = S3Service()
    return _s3_singleton
