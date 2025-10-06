"""Rate limiting service for API endpoints."""

import logging
from typing import Optional

import redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi import FastAPI

from ...config import config

logger = logging.getLogger(__name__)

try:
    if config.REDIS_URL:
        redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
        redis_client.ping()
        storage_uri = config.REDIS_URL
        logger.info(f"Using Redis for rate limiting storage: {config.REDIS_URL}")
    else:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        redis_client.ping()
        storage_uri = "redis://localhost:6379"
        logger.info("Using Redis for rate limiting storage: localhost:6379")
except Exception as e:
    logger.warning(f"Redis not available, using in-memory storage: {e}")
    storage_uri = "memory://"

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=storage_uri,
    default_limits=["1000 per day", "100 per hour"]
)


def setup_rate_limiting(app: FastAPI) -> None:
    """Configure rate limiting for the FastAPI application."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    logger.info("Rate limiting configured")


def get_predict_limit() -> str:
    """Return the prediction rate limit."""
    return f"{config.RATE_LIMIT_PREDICT} per minute"


def get_correction_limit() -> str:
    """Return the correction rate limit."""
    return f"{config.RATE_LIMIT_CORRECTION} per minute"


def get_upload_limit() -> str:
    """Return the upload rate limit."""
    return f"{config.RATE_LIMIT_UPLOAD} per minute"


PREDICT_LIMIT = get_predict_limit()
CORRECTION_LIMIT = get_correction_limit()
UPLOAD_LIMIT = get_upload_limit()
