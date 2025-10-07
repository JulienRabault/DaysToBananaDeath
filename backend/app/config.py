"""Configuration management for the banana classifier API."""

import os
import warnings
from typing import Optional, Set
from pathlib import Path
from pydantic_settings import BaseSettings


try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[CONFIG] Loaded environment from: {env_path}")
    else:
        print("[CONFIG] No .env file found, using system environment variables")
except ImportError:
    print("[CONFIG] python-dotenv not available, using system environment variables")


def _get_env_with_warning(key: str, default: str, warn_on_default: bool = True) -> str:
    """Get environment variable with optional warning when using default value."""
    value = os.getenv(key)
    if value is None:
        if warn_on_default:
            warnings.warn(f"[CONFIG WARNING] Using default value for {key}: '{default}'", UserWarning)
        return default
    return value


def _get_env_bool_with_warning(key: str, default: str, warn_on_default: bool = True) -> bool:
    """Get boolean environment variable with optional warning when using default value."""
    value = os.getenv(key)
    if value is None:
        if warn_on_default:
            warnings.warn(f"[CONFIG WARNING] Using default value for {key}: '{default}'", UserWarning)
        return default.lower() == "true"
    return value.lower() == "true"


def _get_env_int_with_warning(key: str, default: str, warn_on_default: bool = True) -> int:
    """Get integer environment variable with optional warning when using default value."""
    value = os.getenv(key)
    if value is None:
        if warn_on_default:
            warnings.warn(f"[CONFIG WARNING] Using default value for {key}: '{default}'", UserWarning)
        return int(default)
    return int(value)


def _get_env_float_with_warning(key: str, default: str, warn_on_default: bool = True) -> float:
    """Get float environment variable with optional warning when using default value."""
    value = os.getenv(key)
    if value is None:
        if warn_on_default:
            warnings.warn(f"[CONFIG WARNING] Using default value for {key}: '{default}'", UserWarning)
        return float(default)
    return float(value)


class Config(BaseSettings):
    """Configuration class that loads all environment variables."""

    MODEL_IMG_SIZE: int = _get_env_int_with_warning("MODEL_IMG_SIZE", "224")
    MODEL_TYPE: str = _get_env_with_warning("MODEL_TYPE", "resnet50")
    MODEL_FORMAT: str = _get_env_with_warning("MODEL_FORMAT", "ckpt")
    INFERENCE_DEVICE: str = _get_env_with_warning("INFERENCE_DEVICE", "cpu")

    MODEL_LOCAL_PATH: Optional[str] = os.getenv("MODEL_LOCAL_PATH") or None
    MODEL_S3_KEY: Optional[str] = os.getenv("MODEL_S3_KEY") or None
    WANDB_RUN_PATH: Optional[str] = os.getenv("WANDB_RUN_PATH") or None
    WANDB_ARTIFACT: Optional[str] = os.getenv("WANDB_ARTIFACT") or None

    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = _get_env_with_warning("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: Optional[str] = os.getenv("S3_BUCKET_NAME")

    MODEL_TMP_DIR_PARENT: Optional[str] = os.getenv("MODEL_TMP_DIR_PARENT")

    DAYS_EXPECTED_UNRIPE: float = _get_env_float_with_warning("DAYS_EXPECTED_UNRIPE", "6.0")
    DAYS_EXPECTED_RIPE: float = _get_env_float_with_warning("DAYS_EXPECTED_RIPE", "4.0")
    DAYS_EXPECTED_OVERRIPE: float = _get_env_float_with_warning("DAYS_EXPECTED_OVERRIPE", "1.5")
    DAYS_EXPECTED_ROTTEN: float = _get_env_float_with_warning("DAYS_EXPECTED_ROTTEN", "0.0")
    DAYS_EXPECTED_UNKNOWNS: float = _get_env_float_with_warning("DAYS_EXPECTED_UNKNOWNS", "2.0")

    DAYS_UNRIPE_MIN: int = _get_env_int_with_warning("DAYS_UNRIPE_MIN", "5")
    DAYS_RIPE_MIN: int = _get_env_int_with_warning("DAYS_RIPE_MIN", "2")
    DAYS_RIPE_MAX: int = _get_env_int_with_warning("DAYS_RIPE_MAX", "4")
    DAYS_OVERRIPE_MAX: int = _get_env_int_with_warning("DAYS_OVERRIPE_MAX", "1")

    LOG_LEVEL: str = _get_env_with_warning("LOG_LEVEL", "INFO").upper()

    ENABLE_MOCK_PREDICTIONS: bool = _get_env_bool_with_warning("ENABLE_MOCK_PREDICTIONS", "true")

    ENABLE_CORRECTIONS: bool = _get_env_bool_with_warning("ENABLE_CORRECTIONS", "true")

    FRONTEND_ORIGIN: str = _get_env_with_warning("FRONTEND_ORIGIN", "*")

    DATASET_PREFIX: str = _get_env_with_warning("DATASET_PREFIX", "dataset_new")
    CORRECTIONS_PREFIX: str = _get_env_with_warning("CORRECTIONS_PREFIX", "corrections")
    CORRECTION_COUNTER_KEY: str = _get_env_with_warning("CORRECTION_COUNTER_KEY", "metrics/corrections.json")
    CORRECTION_THRESHOLD: int = _get_env_int_with_warning("CORRECTION_THRESHOLD", "1000")
    UPLOADS_PREFIX: str = _get_env_with_warning("UPLOADS_PREFIX", "uploads")
    UPLOAD_PREFIX: str = _get_env_with_warning("UPLOAD_PREFIX", "incoming")

    ALERT_WEBHOOK_URL: Optional[str] = os.getenv("ALERT_WEBHOOK_URL")

    S3_ENDPOINT_URL: Optional[str] = os.getenv("S3_ENDPOINT_URL")

    RECAPTCHA_SECRET_KEY: Optional[str] = os.getenv("RECAPTCHA_SECRET_KEY")
    RECAPTCHA_MIN_SCORE: float = _get_env_float_with_warning("RECAPTCHA_MIN_SCORE", "0.5")
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")

    RATE_LIMIT_PREDICT: int = _get_env_int_with_warning("RATE_LIMIT_PREDICT", "5")
    RATE_LIMIT_CORRECTION: int = _get_env_int_with_warning("RATE_LIMIT_CORRECTION", "10")
    RATE_LIMIT_UPLOAD: int = _get_env_int_with_warning("RATE_LIMIT_UPLOAD", "15")

    PREDICT_LIMIT: str = _get_env_with_warning("PREDICT_LIMIT", "30 per minute")
    CORRECTION_LIMIT: str = _get_env_with_warning("CORRECTION_LIMIT", "10 per minute")
    UPLOAD_LIMIT: str = _get_env_with_warning("UPLOAD_LIMIT", "50 per hour")

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        errors = []

        if not cls.ENABLE_MOCK_PREDICTIONS:
            if not cls.MODEL_LOCAL_PATH and not cls.MODEL_S3_KEY:
                errors.append("Either MODEL_LOCAL_PATH or MODEL_S3_KEY must be set when mock predictions are disabled")

        if cls.MODEL_S3_KEY and not cls.S3_BUCKET_NAME:
            errors.append("S3_BUCKET_NAME is required when using MODEL_S3_KEY")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
            raise ValueError(error_msg)

    @classmethod
    def print_config(cls) -> None:
        """Print current configuration (without sensitive data)."""
        print("[CONFIG] Current configuration:")
        print(f"  Model format: {cls.MODEL_FORMAT}")
        print(f"  Model type: {cls.MODEL_TYPE}")
        print(f"  Image size: {cls.MODEL_IMG_SIZE}")
        print(f"  Device: {cls.INFERENCE_DEVICE}")
        print(f"  Log level: {cls.LOG_LEVEL}")
        print(f"  Mock predictions: {cls.ENABLE_MOCK_PREDICTIONS}")
        print(f"  Corrections enabled: {cls.ENABLE_CORRECTIONS}")
        print(f"  Has local model: {bool(cls.MODEL_LOCAL_PATH)}")
        print(f"  Has S3 model: {bool(cls.MODEL_S3_KEY)}")
        print(f"  S3 bucket: {cls.S3_BUCKET_NAME or 'Not configured'}")

    @classmethod
    def check_unexpected_env_vars(cls) -> None:
        """Check for unexpected environment variables that start with common prefixes."""
        expected_env_vars = {
            "MODEL_IMG_SIZE", "MODEL_TYPE", "MODEL_FORMAT", "INFERENCE_DEVICE",
            "MODEL_LOCAL_PATH", "MODEL_S3_KEY", "WANDB_RUN_PATH", "WANDB_ARTIFACT",
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "S3_BUCKET_NAME",
            "MODEL_TMP_DIR_PARENT",
            "DAYS_EXPECTED_UNRIPE", "DAYS_EXPECTED_RIPE", "DAYS_EXPECTED_OVERRIPE",
            "DAYS_EXPECTED_ROTTEN", "DAYS_EXPECTED_UNKNOWNS",
            "DAYS_UNRIPE_MIN", "DAYS_RIPE_MIN", "DAYS_RIPE_MAX", "DAYS_OVERRIPE_MAX",
            "LOG_LEVEL", "ENABLE_MOCK_PREDICTIONS", "ENABLE_CORRECTIONS", "FRONTEND_ORIGIN",
            "DATASET_PREFIX", "CORRECTIONS_PREFIX", "CORRECTION_COUNTER_KEY", "CORRECTION_THRESHOLD",
            "UPLOADS_PREFIX", "UPLOAD_PREFIX", "ALERT_WEBHOOK_URL", "S3_ENDPOINT_URL",
            "RECAPTCHA_SECRET_KEY", "RECAPTCHA_MIN_SCORE", "REDIS_URL",
            "RATE_LIMIT_PREDICT", "RATE_LIMIT_CORRECTION", "RATE_LIMIT_UPLOAD",
            "PREDICT_LIMIT", "CORRECTION_LIMIT", "UPLOAD_LIMIT"
        }

        prefixes = ["MODEL_", "AWS_", "S3_", "DAYS_", "LOG_", "ENABLE_", "FRONTEND_", "WANDB_"]

        for key in os.environ:
            if any(key.startswith(prefix) for prefix in prefixes):
                if key not in expected_env_vars:
                    warnings.warn(
                        f"[CONFIG WARNING] Unexpected environment variable detected: '{key}' "
                        f"(not in expected configuration keys)",
                        UserWarning
                    )


config = Config()

config.check_unexpected_env_vars()
