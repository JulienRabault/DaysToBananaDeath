#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import onnxruntime as ort
from PIL import Image
import streamlit as st
import random

CLASSES = ["overripe", "ripe", "rotten", "unripe", "unknowns"]

# Base days per class
CLASS_TO_DAYS = {
    "unripe": 5.0,
    "ripe": 2.0,
    "overripe": 0.5,
    "rotten": 0.0,
    "unknowns": 2.0,
}

DEFAULT_WANDB_ARTIFACT = (
    os.environ.get("WANDB_MODEL_ARTIFACT")
    or os.environ.get("DEFAULT_WANDB_ARTIFACT")
    or "jrabault/banana-classification-unknown/onnx:latest"
)


def _setup_transform(img_size: Tuple[int, int] = (224, 224)) -> A.Compose:
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def _to_numpy_chw_float32(tensor_like) -> np.ndarray:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and hasattr(tensor_like, "detach"):
        arr = tensor_like.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor_like)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, 0)
    return arr.astype(np.float32, copy=False)


@st.cache_resource(show_spinner=False)
def load_model_session() -> Dict:
    model_path_env = os.environ.get("MODEL_ONNX_PATH")
    transform = _setup_transform()
    temp_dir_obj: Optional[tempfile.TemporaryDirectory] = None
    model_path: Optional[str] = None

    if model_path_env:
        model_path = model_path_env
    else:
        try:
            import wandb
            wandb.login()
            api = wandb.Api()
            temp_dir_obj = tempfile.TemporaryDirectory(prefix="wandb_artifacts_")
            artifact = api.artifact(DEFAULT_WANDB_ARTIFACT, type="model")
            artifact_dir = artifact.download(root=temp_dir_obj.name)
            onnx_files = list(Path(artifact_dir).glob("*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(f"No .onnx file found in {DEFAULT_WANDB_ARTIFACT}")
            model_path = str(onnx_files[0])
        except Exception as e:
            return {"session": None, "transform": transform, "tempdir": temp_dir_obj, "model_path": model_path, "error": str(e)}

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return {"session": session, "transform": transform, "tempdir": temp_dir_obj, "model_path": model_path, "error": None}


def preprocess_image(image: Image.Image, transform: A.Compose) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_np = np.array(image)
    transformed = transform(image=image_np)
    image_tensor = transformed["image"]
    return _to_numpy_chw_float32(image_tensor)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def expected_days_from_probs(probs: np.ndarray) -> float:
    """Compute expected remaining days from class probabilities with slight randomness."""
    days_vec = np.array([CLASS_TO_DAYS[c] for c in CLASSES], dtype=np.float32)
    expected = float(np.dot(probs, days_vec))
    # Add small random variation ±1 day for ripe/overripe/unripe
    top_idx = int(np.argmax(probs))
    top_class = CLASSES[top_idx]
    if top_class in ["unripe", "ripe", "overripe"]:
        expected += random.uniform(-1.0, 1.0)
    # Clamp between 0 and 7 days for safety
    return max(0.0, min(7.0, expected))


def predict_days(image: Image.Image, session: ort.InferenceSession, transform: A.Compose) -> Dict:
    batch = preprocess_image(image, transform)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch})
    logits = outputs[0][0]
    probs = softmax(logits)
    days = expected_days_from_probs(probs)
    confidence = float(np.max(probs))
    return {"days_left": days, "confidence": confidence}


# ------------- UI -------------
st.set_page_config(page_title="DayToBananaDeath", page_icon="🍌", layout="centered")

st.title("🍌 Day to banana death")
st.caption("Estimate how many days remain before your banana is no longer edible")

state = load_model_session()

uploaded = st.file_uploader("Upload a banana image", type=["jpg", "jpeg", "png", "webp", "bmp"])
use_camera = st.toggle("Use camera (experimental)", value=False)
camera_image = st.camera_input("Take a photo") if use_camera else None
img_file = camera_image or uploaded

if img_file is not None:
    image = Image.open(img_file)
    st.image(image, caption="Uploaded image", use_container_width=True)
    if state.get("session") is None:
        st.warning("Model not available. Check configuration.")
    else:
        with st.spinner("Predicting..."):
            try:
                res = predict_days(image, state["session"], state["transform"])
                days_left = res["days_left"]
                conf = res["confidence"]

                rounded = max(0.0, round(days_left * 2) / 2.0)
                label = "day" if abs(rounded - 1.0) < 1e-6 else "days"
                st.metric("Estimated remaining days", f"~ {rounded} {label}")
                st.caption(f"Confidence: {conf*100:.0f}%")
            except Exception as e:
                st.error(f"Inference error: {e}")
else:
    st.info("Upload an image or use the camera to start.")

st.divider()

if __name__ == "__main__":
    import webbrowser
    import subprocess

    port = int(os.environ.get("PORT", 8501))
    url = f"http://localhost:{port}"
    try:
        webbrowser.open_new_tab(url)
    except Exception:
        pass
    subprocess.run(["streamlit", "run", __file__, "--server.port", str(port), "--server.address", "0.0.0.0"])
