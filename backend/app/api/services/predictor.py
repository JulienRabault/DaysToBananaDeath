import io
import os
import random
import sys
import tempfile
import shutil
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort  # optional
except Exception:
    ort = None

from .s3 import get_s3


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_class_to_days_mapping() -> Dict[str, float]:
    """Retourne le mapping classe -> jours attendus depuis les variables d'environnement."""
    return {
        "unripe": float(os.getenv("DAYS_EXPECTED_UNRIPE", "6.0")),
        "ripe": float(os.getenv("DAYS_EXPECTED_RIPE", "4.0")),
        "overripe": float(os.getenv("DAYS_EXPECTED_OVERRIPE", "1.5")),
        "rotten": float(os.getenv("DAYS_EXPECTED_ROTTEN", "0.0")),
        "unknowns": float(os.getenv("DAYS_EXPECTED_UNKNOWNS", "2.0")),
    }


class PredictorService:
    def __init__(self):
        # Classes alignées avec l’entraînement
        self.class_names = ['overripe', 'ripe', 'rotten', 'unripe', 'unknowns']
        self.img_size = int(os.getenv("MODEL_IMG_SIZE", "224"))
        self.model_type = os.getenv("MODEL_TYPE", "resnet50")
        self.device = os.getenv("INFERENCE_DEVICE", "cpu")

        # Sources possibles
        self.use_onnx = os.getenv("MODEL_FORMAT", "ckpt").lower() == "onnx"
        self.local_model_path = os.getenv("MODEL_LOCAL_PATH")  # chemin local vers .onnx ou .ckpt
        self.s3_model_key = os.getenv("MODEL_S3_KEY")  # clé S3 vers .onnx ou .ckpt
        self.wandb_run_path = os.getenv("WANDB_RUN_PATH")
        self.wandb_artifact = os.getenv("WANDB_ARTIFACT")

        self._loaded = False
        self._onnx_session: Optional[Any] = None
        self._pt_infer = None
        self._tempdir: Optional[str] = None

    def _ensure_ml_path(self):
        # Ajouter ml/src au PYTHONPATH pour importer la classe d’inférence
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../ml/src"))
        if repo_root not in sys.path:
            sys.path.append(repo_root)

    def _download_model_from_s3_to(self, dest_path: str) -> str:
        if not self.s3_model_key:
            raise RuntimeError("MODEL_S3_KEY is not set")
        s3 = get_s3()
        # Télécharge directement le fichier vers dest_path
        s3.s3.download_file(Bucket=s3.bucket, Key=self.s3_model_key, Filename=dest_path)
        return dest_path

    def _load_pytorch_inference(self):
        self._ensure_ml_path()
        from inference import BananaClassifierInference  # type: ignore
        # Choisir la source
        checkpoint_path = None
        if self.local_model_path and self.local_model_path.endswith('.ckpt'):
            checkpoint_path = self.local_model_path
        elif self.s3_model_key and self.s3_model_key.endswith('.ckpt'):
            # Si nous avons déjà téléchargé dans un tempdir via prepare_startup, self.local_model_path le contiendra
            if self.local_model_path and os.path.exists(self.local_model_path):
                checkpoint_path = self.local_model_path
            else:
                # fallback direct
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.ckpt')
                tmp.close()
                checkpoint_path = self._download_model_from_s3_to(tmp.name)

        if checkpoint_path:
            self._pt_infer = BananaClassifierInference(
                checkpoint_path=checkpoint_path,
                model_type=self.model_type,
                device=self.device,
                img_size=(self.img_size, self.img_size),
            )
        # elif self.wandb_run_path or self.wandb_artifact:
        #     self._pt_infer = BananaClassifierInference(
        #         wandb_run_path=self.wandb_run_path,
        #         wandb_artifact_name=self.wandb_artifact,
        #         model_type=self.model_type,
        #         device=self.device,
        #         img_size=(self.img_size, self.img_size),
        #     )
        else:
            raise RuntimeError("No valid model source provided for PyTorch (.ckpt). Set MODEL_LOCAL_PATH or MODEL_S3_KEY or WANDB_* env vars.")

    def _load_onnx(self):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed. Install onnxruntime or use MODEL_FORMAT=ckpt")
        model_path = None
        if self.local_model_path and self.local_model_path.endswith('.onnx'):
            model_path = self.local_model_path
        elif self.s3_model_key and self.s3_model_key.endswith('.onnx'):
            if self.local_model_path and os.path.exists(self.local_model_path):
                model_path = self.local_model_path
            else:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.onnx')
                tmp.close()
                model_path = self._download_model_from_s3_to(tmp.name)
        if not model_path:
            raise RuntimeError("No valid ONNX model provided. Set MODEL_LOCAL_PATH or MODEL_S3_KEY pointing to .onnx")
        self._onnx_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])  # Railway CPU

    def load(self):
        if self._loaded:
            return
        if self.use_onnx:
            self._load_onnx()
        else:
            self._load_pytorch_inference()
        self._loaded = True

    def prepare_startup(self):
        """Crée un répertoire temporaire pour le modèle, télécharge depuis S3 et force un rechargement."""
        if not self.s3_model_key:
            return  # rien à faire si modèle local ou W&B
        # Nettoyer précédent si présent
        if self._tempdir and os.path.isdir(self._tempdir):
            shutil.rmtree(self._tempdir, ignore_errors=True)
            self._tempdir = None
        parent = os.getenv("MODEL_TMP_DIR_PARENT")
        self._tempdir = tempfile.mkdtemp(prefix="model_", dir=parent if parent else None)
        basename = os.path.basename(self.s3_model_key)
        dest_path = os.path.join(self._tempdir, basename)
        self._download_model_from_s3_to(dest_path)
        # Pointe vers ce fichier
        self.local_model_path = dest_path
        # Forcer reload
        self._loaded = False
        self._onnx_session = None
        self._pt_infer = None
        self.load()

    def cleanup(self):
        """Supprime le répertoire temporaire de modèle si existant."""
        if self._tempdir and os.path.isdir(self._tempdir):
            shutil.rmtree(self._tempdir, ignore_errors=True)
        self._tempdir = None

    def _preprocess_pil(self, img: Image.Image) -> np.ndarray:
        img = img.convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)  # NCHW
        return arr

    def predict_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        self.load()
        pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if self.use_onnx and self._onnx_session is not None:
            inp = self._preprocess_pil(pil)
            input_name = self._onnx_session.get_inputs()[0].name
            outputs = self._onnx_session.run(None, {input_name: inp})
            logits = outputs[0]
            # softmax
            exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exps / np.sum(exps, axis=1, keepdims=True)
            idx = int(np.argmax(probs[0]))
            confidence = float(probs[0, idx])
            class_prob = {self.class_names[i]: float(probs[0, i]) for i in range(len(self.class_names))}
            return {
                "predicted_class": self.class_names[idx],
                "confidence": confidence,
                "predicted_index": idx,
                "class_probabilities": class_prob,
                "expected_days": self._expected_days_from_probs(probs),
                "model_info": {"format": "onnx", "img_size": self.img_size, "model_type": self.model_type},
            }
        else:
            # PyTorch
            result = self._pt_infer.predict_single(pil, return_top_k=5)
            return result

    def predict_from_s3_key(self, key: str) -> Dict[str, Any]:
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
        # Add small random variation ±0.5 day for ripe/overripe/unripe
        top_idx = int(np.argmax(probs))
        top_class = self.class_names[top_idx]
        if top_class in ["unripe", "ripe", "overripe"]:
            expected += random.uniform(-0.5, 0.5)
        # Clamp between 0 and 7 days for safety
        return max(0.0, min(7.0, expected))


_predictor_singleton: Optional[PredictorService] = None

def get_predictor() -> PredictorService:
    global _predictor_singleton
    if _predictor_singleton is None:
        _predictor_singleton = PredictorService()
    return _predictor_singleton
