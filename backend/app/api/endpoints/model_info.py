"""
Model information endpoint.
Provides real model metadata and configuration.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter

from ..services.predictor import get_predictor
from ...config import config

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/model/info")
async def get_model_info() -> Dict[str, Any]:
    """
    Get real model information and configuration.
    Returns actual metadata instead of hardcoded values.
    """
    try:
        predictor = get_predictor()
        
        # Get ONNX model metadata if available
        model_metadata = {}
        version = "unknown"
        
        if predictor.use_onnx and predictor._onnx_session is not None:
            try:
                # Get ONNX model metadata
                session_meta = predictor._onnx_session.get_modelmeta()
                if hasattr(session_meta, 'custom_metadata_map'):
                    model_metadata = dict(session_meta.custom_metadata_map)
                if hasattr(session_meta, 'version'):
                    version = str(session_meta.version)
                elif hasattr(session_meta, 'graph_name'):
                    version = session_meta.graph_name or "unknown"
            except Exception as e:
                logger.warning(f"Could not retrieve ONNX metadata: {e}")
        
        # Build response with real configuration data
        model_info = {
            "name": "Banana Classification Model",
            "version": version,
            "format": "ONNX" if predictor.use_onnx else "PyTorch",
            "imageSize": f"{config.MODEL_IMG_SIZE}x{config.MODEL_IMG_SIZE}",
            "modelType": config.MODEL_TYPE,
            "device": config.INFERENCE_DEVICE,
            "classes": predictor.class_names,
            "classMappings": {
                "days": predictor.get_class_to_days_mapping() if hasattr(predictor, 'get_class_to_days_mapping') else {},
                "thresholds": predictor.get_days_to_class_thresholds() if hasattr(predictor, 'get_days_to_class_thresholds') else {}
            },
            "modelSources": {
                "localPath": config.MODEL_LOCAL_PATH,
                "s3Key": config.MODEL_S3_KEY,
                "wandbPath": config.WANDB_RUN_PATH,
                "wandbArtifact": config.WANDB_ARTIFACT
            },
            "configuration": {
                "mockPredictions": config.ENABLE_MOCK_PREDICTIONS,
                "logLevel": config.LOG_LEVEL,
                "s3Bucket": config.S3_BUCKET_NAME,
                "awsRegion": config.AWS_REGION
            },
            "modelLoaded": predictor._loaded,
            "canLoadModel": predictor._can_load_model(),
            "modelMetadata": model_metadata
        }
        
        # Add timestamp if model is loaded
        if predictor._loaded:
            import time
            model_info["loadedAt"] = time.time()
        
        logger.info(f"[MODEL_INFO] Returning model information: format={model_info['format']}, loaded={model_info['modelLoaded']}")
        
        return model_info
        
    except Exception as e:
        logger.error(f"[MODEL_INFO] Error retrieving model info: {e}")
        # Return minimal info if there's an error
        return {
            "name": "Banana Classification Model",
            "version": "unknown",
            "format": config.MODEL_FORMAT.upper(),
            "imageSize": f"{config.MODEL_IMG_SIZE}x{config.MODEL_IMG_SIZE}",
            "modelType": config.MODEL_TYPE,
            "device": config.INFERENCE_DEVICE,
            "classes": ["unripe", "ripe", "overripe", "rotten", "unknowns"],
            "error": str(e),
            "modelLoaded": False
        }
