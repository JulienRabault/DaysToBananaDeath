"""FastAPI application for banana ripeness prediction."""

from typing import Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.endpoints.predict import router as predict_router
from .api.endpoints.presign import router as presign_router
from .api.endpoints.correction import router as correction_router
from .api.endpoints.model_info import router as model_info_router
from .api.services.predictor import get_predictor
from .api.services.rate_limiter import setup_rate_limiting
from .config import config

app = FastAPI(title="Banana Ripeness API", version="0.1.0")

# Configuration CORS pour supporter plusieurs domaines
allowed_origins = []
if config.FRONTEND_ORIGIN == "*":
    allowed_origins = ["*"]
else:
    # Support pour plusieurs domaines séparés par des virgules
    origins = [origin.strip() for origin in config.FRONTEND_ORIGIN.split(",")]
    allowed_origins = origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_rate_limiting(app)

app.include_router(predict_router, prefix="/api")
app.include_router(presign_router, prefix="/api")
app.include_router(correction_router, prefix="/api")
app.include_router(model_info_router, prefix="/api")


@app.on_event("startup")
async def _startup() -> None:
    """Download model from S3 to temp directory if configured."""
    try:
        predictor = get_predictor()
        predictor.prepare_startup()
    except Exception as e:
        print(f"[startup] Predictor prepare_startup error: {e}")


@app.on_event("shutdown")
async def _shutdown() -> None:
    """Cleanup predictor resources."""
    try:
        predictor = get_predictor()
        predictor.cleanup()
    except Exception as e:
        print(f"[shutdown] Predictor cleanup error: {e}")


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {"name": "Banana Ripeness API", "version": "0.1.0"}
