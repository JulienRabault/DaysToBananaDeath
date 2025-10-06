"""Service for banana ripeness prediction using ML models."""

import io
import os
import random
import sys
import tempfile
import shutil
import logging
import time
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from ...config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

DEBUG_MODE = config.LOG_LEVEL == "DEBUG"

try:
    import onnxruntime as ort
except Exception:
    ort = None

from .s3 import get_s3


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_class_to_days_mapping() -> Dict[str, float]:
    """Returns the class -> expected days mapping from configuration."""
    return {
        "unripe": config.DAYS_EXPECTED_UNRIPE,
        "ripe": config.DAYS_EXPECTED_RIPE,
        "overripe": config.DAYS_EXPECTED_OVERRIPE,
        "rotten": config.DAYS_EXPECTED_ROTTEN,
        "unknowns": config.DAYS_EXPECTED_UNKNOWNS,
    }


def get_days_to_class_thresholds() -> Dict[str, int]:
    """Returns the thresholds for mapping days_left to class."""
    return {
        "unripe_min": config.DAYS_UNRIPE_MIN,
        "ripe_min": config.DAYS_RIPE_MIN,
        "ripe_max": config.DAYS_RIPE_MAX,
        "overripe_max": config.DAYS_OVERRIPE_MAX,
    }


def map_days_left_to_class(days_left: float) -> str:
    """Maps remaining days to banana ripeness class based on thresholds."""
    thresholds = get_days_to_class_thresholds()

    days = int(round(days_left))

    if days >= thresholds["unripe_min"]:
        return "unripe"
    elif thresholds["ripe_min"] <= days <= thresholds["ripe_max"]:
        return "ripe"
    elif 1 <= days <= thresholds["overripe_max"]:
        return "overripe"
    elif days == 0:
        return "rotten"
    else:
        return "unknowns"


class PredictorService:
    """Service for banana ripeness prediction using ML models."""

    def __init__(self) -> None:
        logger.info("[PREDICTOR] Initializing PredictorService")

        self.class_names = ['overripe', 'ripe', 'rotten', 'unripe', 'unknowns']
        self.img_size = config.MODEL_IMG_SIZE
        self.model_type = config.MODEL_TYPE
        self.device = config.INFERENCE_DEVICE

        self.use_onnx = config.MODEL_FORMAT.lower() == "onnx"
        self.local_model_path = config.MODEL_LOCAL_PATH
        self.s3_model_key = config.MODEL_S3_KEY
        self.wandb_run_path = config.WANDB_RUN_PATH
        self.wandb_artifact = config.WANDB_ARTIFACT

        self.enable_mock_predictions = config.ENABLE_MOCK_PREDICTIONS

        logger.info("[PREDICTOR] Configuration:")
        logger.info("   Image size: %s", self.img_size)
        logger.info("   Model type: %s", self.model_type)
        logger.info("   Device: %s", self.device)
        logger.info("   Format: %s", 'ONNX' if self.use_onnx else 'PyTorch')
        logger.info("   Local path: %s", self.local_model_path or 'Not defined')
        logger.info("   S3 key: %s", self.s3_model_key or 'Not defined')
        logger.info("   Mock predictions enabled: %s", self.enable_mock_predictions)

        if DEBUG_MODE:
            logger.debug("[PREDICTOR] Debug logging enabled")

        self._loaded = False
        self._onnx_session: Optional[Any] = None
        self._pt_infer: Optional[Any] = None
        self._tempdir: Optional[str] = None

    def _ensure_ml_path(self) -> None:
        """Add ml/src to PYTHONPATH to import inference class."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../ml/src"))
        if repo_root not in sys.path:
            sys.path.append(repo_root)
            if DEBUG_MODE:
                logger.debug("[PREDICTOR] Added ml/src to Python path: %s", repo_root)

    def _download_model_from_s3_to(self, dest_path: str) -> str:
        """Download model from S3 to specified path."""
        if not self.s3_model_key:
            raise RuntimeError("MODEL_S3_KEY is not set")
        logger.info("[PREDICTOR] Downloading model from S3 to: %s", dest_path)
        s3 = get_s3()
        s3.s3.download_file(Bucket=s3.bucket, Key=self.s3_model_key, Filename=dest_path)
        return dest_path

    def _load_pytorch_inference(self) -> None:
        """Load PyTorch model for inference."""
        logger.info("[PREDICTOR] Loading PyTorch model...")
        start_time = time.time()

        self._ensure_ml_path()
        from inference import BananaClassifierInference

        checkpoint_path = None
        if self.local_model_path and self.local_model_path.endswith('.ckpt'):
            logger.info("[PREDICTOR] Using local model: %s", self.local_model_path)
            checkpoint_path = self.local_model_path
        elif self.s3_model_key and self.s3_model_key.endswith('.ckpt'):
            logger.info("[PREDICTOR] Downloading from S3: %s", self.s3_model_key)
            if self.local_model_path and os.path.exists(self.local_model_path):
                checkpoint_path = self.local_model_path
            else:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.ckpt')
                tmp.close()
                checkpoint_path = self._download_model_from_s3_to(tmp.name)
                logger.info("[PREDICTOR] Model downloaded to: %s", checkpoint_path)

        if checkpoint_path:
            logger.info("[PREDICTOR] Initializing model from: %s", checkpoint_path)
            if DEBUG_MODE:
                logger.debug("[PREDICTOR] Model parameters - type: %s, device: %s, size: %s",
                           self.model_type, self.device, self.img_size)

            self._pt_infer = BananaClassifierInference(
                checkpoint_path=checkpoint_path,
                model_type=self.model_type,
                device=self.device,
                img_size=(self.img_size, self.img_size),
            )
            load_time = (time.time() - start_time) * 1000
            logger.info("[PREDICTOR] PyTorch model loaded in %.2fms", load_time)
        else:
            logger.error("[PREDICTOR] No valid PyTorch model source found")
            raise RuntimeError("No valid model source provided for PyTorch (.ckpt). Set MODEL_LOCAL_PATH or MODEL_S3_KEY or WANDB_* env vars.")

    def _load_onnx(self) -> None:
        """Load ONNX model for inference."""
        logger.info("[PREDICTOR] Loading ONNX model...")
        start_time = time.time()

        if ort is None:
            logger.error("[PREDICTOR] onnxruntime is not installed")
            raise RuntimeError("onnxruntime is not installed. Install onnxruntime or use MODEL_FORMAT=ckpt")

        model_path = None
        if self.local_model_path and self.local_model_path.endswith('.onnx'):
            logger.info("[PREDICTOR] Using local ONNX model: %s", self.local_model_path)
            model_path = self.local_model_path
        elif self.s3_model_key and self.s3_model_key.endswith('.onnx'):
            logger.info("[PREDICTOR] Downloading ONNX from S3: %s", self.s3_model_key)
            if self.local_model_path and os.path.exists(self.local_model_path):
                model_path = self.local_model_path
            else:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.onnx')
                tmp.close()
                model_path = self._download_model_from_s3_to(tmp.name)
                logger.info("[PREDICTOR] ONNX model downloaded to: %s", model_path)

        if not model_path:
            logger.error("[PREDICTOR] No valid ONNX model found")
            raise RuntimeError("No valid ONNX model provided. Set MODEL_LOCAL_PATH or MODEL_S3_KEY pointing to .onnx")

        logger.info("[PREDICTOR] Creating ONNX session...")
        if DEBUG_MODE:
            logger.debug("[PREDICTOR] Using CPUExecutionProvider for ONNX")

        self._onnx_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        load_time = (time.time() - start_time) * 1000
        logger.info("[PREDICTOR] ONNX model loaded in %.2fms", load_time)

    def load(self) -> None:
        """Load the model if not already loaded."""
        if self._loaded:
            if DEBUG_MODE:
                logger.debug("[PREDICTOR] Model already loaded, skipping")
            return
        if self.use_onnx:
            self._load_onnx()
        else:
            self._load_pytorch_inference()
        self._loaded = True

    def prepare_startup(self) -> None:
        """Creates temporary directory for model, downloads from S3 and forces reload."""
        if not self.s3_model_key:
            if DEBUG_MODE:
                logger.debug("[PREDICTOR] No S3 key configured, skipping startup preparation")
            return

        logger.info("[PREDICTOR] Preparing startup with S3 model")

        try:
            if self._tempdir and os.path.isdir(self._tempdir):
                shutil.rmtree(self._tempdir, ignore_errors=True)
                self._tempdir = None

            parent = config.MODEL_TMP_DIR_PARENT
            if not parent:
                parent = tempfile.gettempdir()
                if DEBUG_MODE:
                    logger.debug("[PREDICTOR] Using automatic temp directory: %s", parent)
            else:
                if DEBUG_MODE:
                    logger.debug("[PREDICTOR] Using configured temp directory: %s", parent)

            self._tempdir = tempfile.mkdtemp(prefix="model_", dir=parent)
            basename = os.path.basename(self.s3_model_key)
            dest_path = os.path.join(self._tempdir, basename)

            if DEBUG_MODE:
                logger.debug("[PREDICTOR] Created temp directory: %s", self._tempdir)

            self._download_model_from_s3_to(dest_path)
            self.local_model_path = dest_path
            self._loaded = False
            self._onnx_session = None
            self._pt_infer = None
            self.load()

        except Exception as e:
            logger.error("[PREDICTOR] Failed to download model from S3: %s", str(e))
            logger.error("[PREDICTOR] S3 configuration issues detected:")
            logger.error("   - Check if S3_BUCKET_NAME is set: %s", config.S3_BUCKET_NAME or "NOT SET")
            logger.error("   - Check if AWS_ACCESS_KEY_ID is set: %s", "SET" if config.AWS_ACCESS_KEY_ID else "NOT SET")
            logger.error("   - Check if AWS_SECRET_ACCESS_KEY is set: %s", "SET" if config.AWS_SECRET_ACCESS_KEY else "NOT SET")
            logger.error("   - S3 key: %s", self.s3_model_key)

            if not self.enable_mock_predictions:
                logger.warning("[PREDICTOR] Automatically enabling mock predictions due to S3 error")
                self.enable_mock_predictions = True

            if self._tempdir and os.path.isdir(self._tempdir):
                shutil.rmtree(self._tempdir, ignore_errors=True)
                self._tempdir = None

    def cleanup(self) -> None:
        """Removes temporary model directory if it exists."""
        if self._tempdir and os.path.isdir(self._tempdir):
            logger.info("[PREDICTOR] Cleaning up temporary directory: %s", self._tempdir)
            shutil.rmtree(self._tempdir, ignore_errors=True)
        self._tempdir = None

    def _preprocess_pil(self, img: Image.Image) -> np.ndarray:
        """Preprocess PIL image for model inference."""
        if DEBUG_MODE:
            logger.debug("[PREDICTOR] Preprocessing image from %s to %dx%d", img.size, self.img_size, self.img_size)

        img = img.convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)

        if DEBUG_MODE:
            logger.debug("[PREDICTOR] Preprocessed tensor shape: %s", arr.shape)

        return arr

    def predict_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """Predict banana ripeness from image bytes."""
        start_time = time.time()
        logger.info("[PREDICTOR] Starting prediction (image size: %d bytes)", len(image_bytes))

        if not self._loaded and not self._can_load_model():
            if self.enable_mock_predictions:
                logger.info("[PREDICTOR] Test mode activated - Generating mock prediction")
                result = self._get_mock_prediction()
                prediction_time = (time.time() - start_time) * 1000
                logger.info("[PREDICTOR] Mock prediction generated in %.2fms", prediction_time)
                return result
            else:
                logger.error("[PREDICTOR] No model available and mock predictions disabled")
                raise RuntimeError("No model available and mock predictions are disabled")

        logger.info("[PREDICTOR] Loading model if necessary...")
        self.load()

        logger.info("[PREDICTOR] Preprocessing image...")
        pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        logger.info("   Original size: %s", pil.size)

        if self.use_onnx and self._onnx_session is not None:
            logger.info("[PREDICTOR] Using ONNX model...")
            inference_start = time.time()

            inp = self._preprocess_pil(pil)
            if DEBUG_MODE:
                logger.debug("   Preprocessed tensor: %s", inp.shape)

            input_name = self._onnx_session.get_inputs()[0].name
            if DEBUG_MODE:
                logger.debug("   ONNX input name: %s", input_name)

            outputs = self._onnx_session.run(None, {input_name: inp})
            logits = outputs[0]

            exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exps / np.sum(exps, axis=1, keepdims=True)
            idx = int(np.argmax(probs[0]))
            confidence = float(probs[0, idx])
            predicted_class = self.class_names[idx]

            inference_time = (time.time() - inference_start) * 1000
            logger.info("   ONNX inference: %.2fms", inference_time)

            if DEBUG_MODE:
                logger.debug("   Raw logits shape: %s", logits.shape)
                logger.debug("   Softmax probabilities: %s", probs[0])

            class_prob = {self.class_names[i]: float(probs[0, i]) for i in range(len(self.class_names))}
            expected_days = self._expected_days_from_probs(probs)

            result = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "predicted_index": idx,
                "class_probabilities": class_prob,
                "expected_days": expected_days,
                "model_info": {"format": "onnx", "img_size": self.img_size, "model_type": self.model_type},
            }
        else:
            logger.info("[PREDICTOR] Using PyTorch model...")
            inference_start = time.time()

            result = self._pt_infer.predict_single(pil, return_top_k=5)

            inference_time = (time.time() - inference_start) * 1000
            logger.info("   PyTorch inference: %.2fms", inference_time)
            predicted_class = result.get("predicted_class", "unknown")
            confidence = result.get("confidence", 0.0)

        total_time = (time.time() - start_time) * 1000
        logger.info("[PREDICTOR] Prediction completed in %.2fms", total_time)
        logger.info("[PREDICTOR] Result: %s (confidence: %.2f%%)", predicted_class, confidence * 100)

        return result

    def predict_from_s3_key(self, key: str) -> Dict[str, Any]:
        """Predict banana ripeness from S3 stored image."""
        logger.info("[PREDICTOR] Predicting from S3 key: %s", key)
        s3 = get_s3()
        tmp_path = s3.download_to_tempfile(key)
        with open(tmp_path, 'rb') as f:
            bytes_data = f.read()
        return self.predict_image_bytes(bytes_data)

    def _expected_days_from_probs(self, probs: np.ndarray) -> float:
        """Compute expected remaining days from class probabilities with slight randomness."""
        class_to_days = get_class_to_days_mapping()
        days_vec = np.array([class_to_days[c] for c in self.class_names], dtype=np.float32)
        expected = float(np.dot(probs, days_vec))

        if DEBUG_MODE:
            logger.debug("[PREDICTOR] Days mapping: %s", class_to_days)
            logger.debug("[PREDICTOR] Expected days before randomization: %.2f", expected)

        top_idx = int(np.argmax(probs))
        top_class = self.class_names[top_idx]
        if top_class in ["unripe", "ripe", "overripe"]:
            variation = random.uniform(-0.5, 0.5)
            expected += variation
            if DEBUG_MODE:
                logger.debug("[PREDICTOR] Added variation %.2f for class %s", variation, top_class)

        clamped = max(0.0, min(7.0, expected))

        if DEBUG_MODE and clamped != expected:
            logger.debug("[PREDICTOR] Clamped expected days from %.2f to %.2f", expected, clamped)

        return clamped

    def _can_load_model(self) -> bool:
        """Checks if a model can be loaded."""
        can_load = False
        if self.use_onnx:
            can_load = (self.local_model_path and self.local_model_path.endswith('.onnx') and os.path.exists(self.local_model_path)) or \
                      (self.s3_model_key and self.s3_model_key.endswith('.onnx'))
        else:
            can_load = (self.local_model_path and self.local_model_path.endswith('.ckpt') and os.path.exists(self.local_model_path)) or \
                      (self.s3_model_key and self.s3_model_key.endswith('.ckpt'))

        if DEBUG_MODE:
            logger.debug("[PREDICTOR] Can load model: %s", can_load)
            logger.debug("   Local path: %s (exists: %s)",
                        self.local_model_path,
                        os.path.exists(self.local_model_path) if self.local_model_path else False)
            logger.debug("   S3 key: %s", self.s3_model_key)

        return can_load

    def _get_mock_prediction(self) -> Dict[str, Any]:
        """Returns a mock prediction for testing."""
        if DEBUG_MODE:
            logger.debug("[PREDICTOR] Generating mock prediction")

        predicted_class = random.choice(self.class_names)
        confidence = random.uniform(0.6, 0.95)
        predicted_index = self.class_names.index(predicted_class)

        if DEBUG_MODE:
            logger.debug("   Mock class: %s, confidence: %.2f", predicted_class, confidence)

        class_prob = {}
        remaining_prob = 1.0 - confidence
        for i, class_name in enumerate(self.class_names):
            if i == predicted_index:
                class_prob[class_name] = confidence
            else:
                class_prob[class_name] = remaining_prob / (len(self.class_names) - 1)

        class_to_days = get_class_to_days_mapping()
        expected_days = class_to_days.get(predicted_class, 3.0)

        result = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "predicted_index": predicted_index,
            "class_probabilities": class_prob,
            "expected_days": expected_days,
            "model_info": {"format": "mock", "img_size": self.img_size, "model_type": "test"},
            "warning": "Test mode - Mock prediction (no model configured)"
        }

        if DEBUG_MODE:
            logger.debug("[PREDICTOR] Mock prediction result: %s", result)

        return result

    def get_class_from_days_left(self, days_left: float) -> str:
        """Returns the appropriate class based on days left using thresholds."""
        mapped_class = map_days_left_to_class(days_left)

        if DEBUG_MODE:
            logger.debug("[PREDICTOR] Mapped %.1f days to class: %s", days_left, mapped_class)
            thresholds = get_days_to_class_thresholds()
            logger.debug("   Using thresholds: %s", thresholds)

        return mapped_class

    def validate_correction(self, days_left: float, user_class: str) -> Dict[str, Any]:
        """Validates a user correction by comparing with threshold-based mapping."""
        suggested_class = self.get_class_from_days_left(days_left)
        is_consistent = suggested_class == user_class

        validation_result = {
            "user_class": user_class,
            "suggested_class": suggested_class,
            "days_left": days_left,
            "is_consistent": is_consistent,
            "confidence_score": 1.0 if is_consistent else 0.5
        }

        if DEBUG_MODE:
            logger.debug("[PREDICTOR] Correction validation: %s", validation_result)

        logger.info("[PREDICTOR] User correction: %.1f days -> %s (suggested: %s, consistent: %s)",
                   days_left, user_class, suggested_class, is_consistent)

        return validation_result

    def get_class_to_days_mapping(self) -> Dict[str, float]:
        """Expose the class to days mapping from configuration."""
        return get_class_to_days_mapping()

    def get_days_to_class_thresholds(self) -> Dict[str, int]:
        """Expose the thresholds for mapping days_left to class."""
        return get_days_to_class_thresholds()


_predictor_singleton: Optional[PredictorService] = None


def get_predictor() -> PredictorService:
    """Get the global predictor service instance."""
    global _predictor_singleton
    if _predictor_singleton is None:
        _predictor_singleton = PredictorService()
    return _predictor_singleton
