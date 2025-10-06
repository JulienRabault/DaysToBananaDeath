"""reCAPTCHA verification service."""

import logging
from typing import Optional, Dict, Any

import httpx

from ...config import config

logger = logging.getLogger(__name__)


class RecaptchaError(Exception):
    """Exception raised during reCAPTCHA validation errors."""
    pass


async def verify_recaptcha_token(token: str, action: str = "correction") -> Dict[str, Any]:
    """Verify a reCAPTCHA v3 token with Google.

    Args:
        token: The reCAPTCHA token received from frontend
        action: The expected action (default "correction")

    Returns:
        Dict containing verification results

    Raises:
        RecaptchaError: If verification fails
    """
    if not config.RECAPTCHA_SECRET_KEY:
        logger.warning("RECAPTCHA_SECRET_KEY not configured, skipping verification")
        return {"success": True, "score": 1.0, "action": action, "skipped": True}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://www.google.com/recaptcha/api/siteverify",
                data={
                    "secret": config.RECAPTCHA_SECRET_KEY,
                    "response": token
                },
                timeout=10.0
            )
            
            if response.status_code != 200:
                raise RecaptchaError(f"reCAPTCHA API returned status {response.status_code}")
            
            result = response.json()
            
            if not result.get("success", False):
                error_codes = result.get("error-codes", [])
                logger.warning(f"reCAPTCHA verification failed: {error_codes}")
                raise RecaptchaError(f"reCAPTCHA verification failed: {', '.join(error_codes)}")

            score = result.get("score", 0.0)
            min_score = getattr(config, 'RECAPTCHA_MIN_SCORE', 0.5)
            
            if score < min_score:
                logger.warning(f"reCAPTCHA score too low: {score} < {min_score}")
                raise RecaptchaError(f"Security score insufficient: {score}")

            returned_action = result.get("action", "")
            if returned_action != action:
                logger.warning(f"reCAPTCHA action mismatch: expected '{action}', got '{returned_action}'")
                raise RecaptchaError(f"Invalid reCAPTCHA action")

            logger.info(f"reCAPTCHA verification successful: score={score}, action={returned_action}")
            return result
            
    except httpx.TimeoutException:
        logger.error("reCAPTCHA verification timed out")
        raise RecaptchaError("Timeout during reCAPTCHA verification")
    except httpx.RequestError as e:
        logger.error(f"reCAPTCHA verification request failed: {str(e)}")
        raise RecaptchaError("Network error during reCAPTCHA verification")
    except RecaptchaError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during reCAPTCHA verification: {str(e)}")
        raise RecaptchaError("Unexpected error during reCAPTCHA verification")


def is_recaptcha_enabled() -> bool:
    """Check if reCAPTCHA is configured and enabled."""
    return bool(config.RECAPTCHA_SECRET_KEY)
