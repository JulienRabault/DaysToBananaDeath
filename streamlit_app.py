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

# Chargement .env pour usage local
try:
    from dotenv import load_dotenv
    load_dotenv()  # ignore errors si absent
except Exception:
    pass

CLASSES = ["overripe", "ripe", "rotten", "unripe", "unknowns"]
# Jours restants approximatifs par classe (heuristique)
CLASS_TO_DAYS = {
    "unripe": 5.0,     # verte → plusieurs jours
    "ripe": 2.0,       # mûre → quelques jours
    "overripe": 0.5,   # trop mûre → < 1 jour
    "rotten": 0.0,     # pourrie → 0
    "unknowns": 2.0,   # par défaut neutre
}

# Choix de l'artifact W&B à partir de différentes variables d'env possibles
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
    """Convertit un tenseur albumentations/Torch en numpy (1, C, H, W) float32."""
    # transformed['image'] avec ToTensorV2 retourne un torch.Tensor
    try:
        import torch  # lazy import au cas où
    except Exception:
        torch = None

    if torch is not None and hasattr(tensor_like, "detach"):
        arr = tensor_like.detach().cpu().numpy()
    else:
        # Si jamais c'est déjà un numpy
        arr = np.asarray(tensor_like)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, 0)
    return arr.astype(np.float32, copy=False)


@st.cache_resource(show_spinner=False)
def load_model_session() -> Dict:
    """Charge le modèle ONNX en mémoire et retourne un dict {session, transform, tempdir, model_path}."""
    model_path_env = os.environ.get("MODEL_ONNX_PATH")
    transform = _setup_transform()

    temp_dir_obj: Optional[tempfile.TemporaryDirectory] = None
    model_path: Optional[str] = None

    if model_path_env:
        model_path = model_path_env
    else:
        # Téléchargement via W&B
        try:
            import wandb  # noqa: WPS433

            # On évite d'écrire de runs : juste API
            wandb.login()
            api = wandb.Api()

            temp_dir_obj = tempfile.TemporaryDirectory(prefix="wandb_artifacts_")
            artifact = api.artifact(DEFAULT_WANDB_ARTIFACT, type="model")
            artifact_dir = artifact.download(root=temp_dir_obj.name)

            # Cherche un .onnx dans l'artifact
            onnx_files = list(Path(artifact_dir).glob("*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(
                    f"Aucun fichier .onnx trouvé dans l'artifact {DEFAULT_WANDB_ARTIFACT}"
                )
            model_path = str(onnx_files[0])
        except Exception as e:  # noqa: BLE001
            # En cas d'échec (clé manquante, réseau…), on renvoie une structure sans session
            return {
                "session": None,
                "transform": transform,
                "tempdir": temp_dir_obj,
                "model_path": model_path,
                "error": str(e),
            }

    # Création de la session ONNX Runtime
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)

    return {
        "session": session,
        "transform": transform,
        "tempdir": temp_dir_obj,
        "model_path": model_path,
        "error": None,
    }


def preprocess_image(image: Image.Image, transform: A.Compose) -> np.ndarray:
    # Convertir en RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_np = np.array(image)
    transformed = transform(image=image_np)
    image_tensor = transformed["image"]
    image_batch = _to_numpy_chw_float32(image_tensor)
    return image_batch


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def expected_days_from_probs(probs: np.ndarray) -> float:
    """Calcule l'espérance de jours restants à partir des probabilités par classe."""
    # Aligne les jours selon l'ordre CLASSES
    days_vec = np.array([CLASS_TO_DAYS[c] for c in CLASSES], dtype=np.float32)
    return float(np.dot(probs, days_vec))


def predict_days(image: Image.Image, session: ort.InferenceSession, transform: A.Compose) -> Dict:
    batch = preprocess_image(image, transform)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch})
    logits = outputs[0][0]
    probs = softmax(logits)

    # Estimation en jours (espérance)
    days = expected_days_from_probs(probs)

    # Mesure simple de fiabilité = proba de la classe majoritaire
    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])

    return {
        "days_left": days,
        "confidence": confidence,
    }


# ------------- UI -------------
st.set_page_config(page_title="BananaCheck - Jours restants", page_icon="🍌", layout="centered")

st.title("🍌 BananaCheck")
st.caption("Estimation du nombre de jours restants avant que la banane ne soit impropre à la consommation")

state = load_model_session()

with st.sidebar:
    st.header("Configuration")
    st.write("Modèle ONNX")
    st.code(state.get("model_path") or os.environ.get("MODEL_ONNX_PATH", "(non défini)"), language="text")
    if state.get("error"):
        st.error(
            "Impossible de charger le modèle via W&B/URL.\n"
            "Définissez WANDB_API_KEY + (WANDB_MODEL_ARTIFACT|DEFAULT_WANDB_ARTIFACT),\n"
            "ou MODEL_ONNX_PATH / MODEL_ONNX_URL.\n\n"
            f"Erreur: {state['error']}"
        )
        st.info(
            "Astuce: pour un démarrage rapide sans W&B, placez un fichier .onnx local et définissez\n"
            "MODEL_ONNX_PATH=/chemin/vers/model.onnx (ou MODEL_ONNX_URL=https://.../model.onnx)."
        )
    else:
        st.success("Modèle chargé ✅")

uploaded = st.file_uploader("Chargez une image de banane", type=["jpg", "jpeg", "png", "webp", "bmp"])
# Option caméra (peut ne pas fonctionner selon la plateforme d'hébergement)
st.write("")
use_camera = st.toggle("Utiliser la caméra (expérimental)", value=False)
if use_camera:
    camera_image = st.camera_input("Prenez une photo")
else:
    camera_image = None

img_file = camera_image or uploaded

col1, col2 = st.columns(2)

with col1:
    st.subheader("Image")
    if img_file is not None:
        image = Image.open(img_file)
        st.image(image, caption="Image chargée", use_container_width=True)
    else:
        st.info("Uploadez une image ou utilisez la caméra pour démarrer.")

with col2:
    st.subheader("Estimation")
    if img_file is not None:
        if state.get("session") is None:
            st.warning("Le modèle n'est pas disponible. Vérifiez la configuration (W&B/URL/chemin).")
        else:
            with st.spinner("Calcul en cours..."):
                try:
                    res = predict_days(image, state["session"], state["transform"])
                    days_left = res["days_left"]
                    conf = res["confidence"]

                    # Affichage principal: jours restants (arrondi à 0.5 près)
                    rounded = max(0.0, round(days_left * 2) / 2.0)
                    label = "jour" if abs(rounded - 1.0) < 1e-6 else "jours"
                    st.metric("Jours restants (estimation)", f"~ {rounded} {label}")

                    # Indication de fiabilité simple
                    st.caption(f"Indice de fiabilité: {conf*100:.0f}%")

                except Exception as e:  # noqa: BLE001
                    st.error(f"Erreur pendant l'inférence: {e}")

st.divider()
st.markdown(
    """
    Notes:
    - Estimation dérivée d'un modèle de classification (unripe/ripe/overripe/rotten/unknowns)
      convertie en jours restants via une heuristique simple.
    - Valeurs indicatives: unripe≈5j, ripe≈2j, overripe≈0.5j, rotten≈0j, unknowns≈2j.
    - Pour plus de précision, entraînez un modèle de régression en jours.
    """
)

if __name__ == "__main__":
    # Permet d'exécuter `python streamlit_app.py` pour un test rapide en local
    # (Streamlit recommandera plutôt `streamlit run streamlit_app.py`)
    import webbrowser
    import subprocess

    port = int(os.environ.get("PORT", 8501))
    url = f"http://localhost:{port}"
    try:
        webbrowser.open_new_tab(url)
    except Exception:
        pass
    subprocess.run(["streamlit", "run", __file__, "--server.port", str(port), "--server.address", "0.0.0.0"])
