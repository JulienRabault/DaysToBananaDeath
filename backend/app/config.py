"""
Configuration management for the banana classifier API.
Handles loading environment variables from .env file in development
and from system environment in production.
"""

import os
import warnings
from typing import Optional, Set
from pathlib import Path

# Try to load .env file if it exists (development)
try:
    from dotenv import load_dotenv

    # Look for .env file in project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[CONFIG] Loaded environment from: {env_path}")
    else:
        print("[CONFIG] No .env file found, using system environment variables")
except ImportError:
    # dotenv not installed, use system environment (production)
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


class Config:
    """Configuration class that loads all environment variables."""

    # Define expected environment variables
    EXPECTED_ENV_VARS: Set[str] = {
        "MODEL_IMG_SIZE", "MODEL_TYPE", "MODEL_FORMAT", "INFERENCE_DEVICE",
        "MODEL_LOCAL_PATH", "MODEL_S3_KEY", "WANDB_RUN_PATH", "WANDB_ARTIFACT",
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "S3_BUCKET_NAME",
        "MODEL_TMP_DIR_PARENT",
        "DAYS_EXPECTED_UNRIPE", "DAYS_EXPECTED_RIPE", "DAYS_EXPECTED_OVERRIPE",
        "DAYS_EXPECTED_ROTTEN", "DAYS_EXPECTED_UNKNOWNS",
        "DAYS_UNRIPE_MIN", "DAYS_RIPE_MIN", "DAYS_RIPE_MAX", "DAYS_OVERRIPE_MAX",
        "LOG_LEVEL", "ENABLE_MOCK_PREDICTIONS", "FRONTEND_ORIGIN",
        "DATASET_PREFIX", "CORRECTIONS_PREFIX", "CORRECTION_COUNTER_KEY", "CORRECTION_THRESHOLD",
        "UPLOADS_PREFIX", "UPLOAD_PREFIX", "ALERT_WEBHOOK_URL", "S3_ENDPOINT_URL"
    }

    # Model Configuration
    MODEL_IMG_SIZE: int = _get_env_int_with_warning("MODEL_IMG_SIZE", "224")
    MODEL_TYPE: str = _get_env_with_warning("MODEL_TYPE", "resnet50")
    MODEL_FORMAT: str = _get_env_with_warning("MODEL_FORMAT", "ckpt")
    INFERENCE_DEVICE: str = _get_env_with_warning("INFERENCE_DEVICE", "cpu")

    # Model Sources (no warnings for optional values)
    MODEL_LOCAL_PATH: Optional[str] = os.getenv("MODEL_LOCAL_PATH") or None
    MODEL_S3_KEY: Optional[str] = os.getenv("MODEL_S3_KEY") or None
    WANDB_RUN_PATH: Optional[str] = os.getenv("WANDB_RUN_PATH") or None
    WANDB_ARTIFACT: Optional[str] = os.getenv("WANDB_ARTIFACT") or None

    # S3 Configuration (no warnings for optional AWS credentials)
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = _get_env_with_warning("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: Optional[str] = os.getenv("S3_BUCKET_NAME")

    # Model temporary directory (optional)
    MODEL_TMP_DIR_PARENT: Optional[str] = os.getenv("MODEL_TMP_DIR_PARENT")

    # Days mapping for banana ripeness
    DAYS_EXPECTED_UNRIPE: float = _get_env_float_with_warning("DAYS_EXPECTED_UNRIPE", "6.0")
    DAYS_EXPECTED_RIPE: float = _get_env_float_with_warning("DAYS_EXPECTED_RIPE", "4.0")
    DAYS_EXPECTED_OVERRIPE: float = _get_env_float_with_warning("DAYS_EXPECTED_OVERRIPE", "1.5")
    DAYS_EXPECTED_ROTTEN: float = _get_env_float_with_warning("DAYS_EXPECTED_ROTTEN", "0.0")
    DAYS_EXPECTED_UNKNOWNS: float = _get_env_float_with_warning("DAYS_EXPECTED_UNKNOWNS", "2.0")

    # Days to class thresholds
    DAYS_UNRIPE_MIN: int = _get_env_int_with_warning("DAYS_UNRIPE_MIN", "5")
    DAYS_RIPE_MIN: int = _get_env_int_with_warning("DAYS_RIPE_MIN", "2")
    DAYS_RIPE_MAX: int = _get_env_int_with_warning("DAYS_RIPE_MAX", "4")
    DAYS_OVERRIPE_MAX: int = _get_env_int_with_warning("DAYS_OVERRIPE_MAX", "1")

    # Logging
    LOG_LEVEL: str = _get_env_with_warning("LOG_LEVEL", "INFO").upper()

    # Mock predictions
    ENABLE_MOCK_PREDICTIONS: bool = _get_env_bool_with_warning("ENABLE_MOCK_PREDICTIONS", "true")

    # API Configuration
    FRONTEND_ORIGIN: str = _get_env_with_warning("FRONTEND_ORIGIN", "*")

    # Dataset and Corrections Configuration
    DATASET_PREFIX: str = _get_env_with_warning("DATASET_PREFIX", "dataset_new")
    CORRECTIONS_PREFIX: str = _get_env_with_warning("CORRECTIONS_PREFIX", "corrections")
    CORRECTION_COUNTER_KEY: str = _get_env_with_warning("CORRECTION_COUNTER_KEY", "metrics/corrections.json")
    CORRECTION_THRESHOLD: int = _get_env_int_with_warning("CORRECTION_THRESHOLD", "1000")
    UPLOADS_PREFIX: str = _get_env_with_warning("UPLOADS_PREFIX", "uploads")
    UPLOAD_PREFIX: str = _get_env_with_warning("UPLOAD_PREFIX", "incoming")

    # Webhooks and Alerts
    ALERT_WEBHOOK_URL: Optional[str] = os.getenv("ALERT_WEBHOOK_URL")

    # S3 Configuration additional
    S3_ENDPOINT_URL: Optional[str] = os.getenv("S3_ENDPOINT_URL")

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        errors = []

        # Check if we're using mock predictions or real model
        if not cls.ENABLE_MOCK_PREDICTIONS:
            if not cls.MODEL_LOCAL_PATH and not cls.MODEL_S3_KEY:
                errors.append("Either MODEL_LOCAL_PATH or MODEL_S3_KEY must be set when mock predictions are disabled")

        # Check S3 configuration if S3 is used
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
        print(f"  Has local model: {bool(cls.MODEL_LOCAL_PATH)}")
        print(f"  Has S3 model: {bool(cls.MODEL_S3_KEY)}")
        print(f"  S3 bucket: {cls.S3_BUCKET_NAME or 'Not configured'}")

    @classmethod
    def check_unexpected_env_vars(cls) -> None:
        """Check for unexpected environment variables that start with common prefixes."""
        prefixes = ["MODEL_", "AWS_", "S3_", "DAYS_", "LOG_", "ENABLE_", "FRONTEND_", "WANDB_"]

        for key in os.environ:
            if any(key.startswith(prefix) for prefix in prefixes):
                if key not in cls.EXPECTED_ENV_VARS:
                    warnings.warn(
                        f"[CONFIG WARNING] Unexpected environment variable detected: '{key}' "
                        f"(not in expected configuration keys)",
                        UserWarning
                    )


# Create global config instance
config = Config()

# Check for unexpected environment variables at startup
config.check_unexpected_env_vars()
