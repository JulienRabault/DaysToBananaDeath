import io
import logging
import time
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import onnxruntime as ort
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model_session = None
transform = None
classes = ["overripe", "ripe", "rotten", "unripe", "unknowns"]
temp_dir = None

def download_model_from_wandb():
    """Download ONNX model from W&B artifacts."""
    global temp_dir

    try:
        logger.info("Downloading model from W&B...")
        wandb.login()

        temp_dir = tempfile.TemporaryDirectory(prefix="wandb_artifacts_")
        api = wandb.Api()
        artifact = api.artifact('jrabault/banana-classification-unknown/onnx:v0', type='model')
        artifact_dir = artifact.download(root=temp_dir.name)

        onnx_files = list(Path(artifact_dir).glob("*.onnx"))
        if not onnx_files:
            logger.error(f"No ONNX file found in {artifact_dir}")
            return None

        model_path = str(onnx_files[0])
        logger.info(f"Model loaded: {model_path}")
        wandb.finish()
        return model_path

    except Exception as e:
        logger.error(f"W&B download error: {e}")
        if temp_dir:
            temp_dir.cleanup()
            temp_dir = None
        return None

def setup_preprocessing():
    """Setup image preprocessing transforms."""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and cleanup."""
    global model_session, transform, temp_dir

    try:
        logger.info("Starting API...")
        transform = setup_preprocessing()

        model_path = download_model_from_wandb()
        if model_path:
            model_session = ort.InferenceSession(model_path)
            logger.info("ONNX model loaded successfully!")
        else:
            logger.error("Failed to load model")
    except Exception as e:
        logger.error(f"Startup error: {e}")

    yield

    logger.info("Shutting down API...")
    if temp_dir:
        temp_dir.cleanup()

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model inference."""
    if transform is None:
        raise ValueError("Transforms not initialized")

    image_array = np.array(image)
    transformed = transform(image=image_array)
    image_tensor = transformed['image']
    return np.expand_dims(image_tensor, axis=0)

app = FastAPI(
    title="Banana Classifier API",
    description="AI-powered banana ripeness classification",
    lifespan=lifespan
)

@app.get("/")
async def home():
    return {"message": "Banana Classifier API - Send a banana image!"}

@app.get("/health")
async def health():
    is_ready = model_session is not None
    return {
        "status": "OK" if is_ready else "ERROR",
        "model_loaded": is_ready
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict banana ripeness from uploaded image."""
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Please upload an image!")

    if model_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded!")

    try:
        start_time = time.time()

        # Load and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_batch = preprocess_image(image)

        # Run inference
        input_name = model_session.get_inputs()[0].name
        outputs = model_session.run(None, {input_name: image_batch})
        predictions = outputs[0][0]

        # Calculate probabilities
        exp_preds = np.exp(predictions)
        probabilities = exp_preds / np.sum(exp_preds)

        predicted_idx = np.argmax(probabilities)
        predicted_class = classes[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        processing_time = round((time.time() - start_time) * 1000, 2)

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                classes[i]: float(probabilities[i])
                for i in range(len(classes))
            },
            "processing_time_ms": processing_time
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    import os
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
