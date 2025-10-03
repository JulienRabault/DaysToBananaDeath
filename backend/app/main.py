from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# Routers (imports relatifs)
from .api.endpoints.predict import router as predict_router
from .api.endpoints.presign import router as presign_router
from .api.endpoints.correction import router as correction_router
from .api.services.predictor import get_predictor

app = FastAPI(title="Banana Ripeness API", version="0.1.0")

# CORS config
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict_router, prefix="/api")
app.include_router(presign_router, prefix="/api")
app.include_router(correction_router, prefix="/api")


@app.on_event("startup")
async def _startup():
    # Télécharger le modèle depuis S3 dans un tempdir si configuré
    try:
        predictor = get_predictor()
        predictor.prepare_startup()
    except Exception as e:
        # Lancer quand même l’API; la prédiction échouera jusqu’à résolution
        print(f"[startup] Predictor prepare_startup error: {e}")


@app.on_event("shutdown")
async def _shutdown():
    try:
        predictor = get_predictor()
        predictor.cleanup()
    except Exception as e:
        print(f"[shutdown] Predictor cleanup error: {e}")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# Optional: root
@app.get("/")
async def root() -> dict:
    return {"name": "Banana Ripeness API", "version": "0.1.0"}
