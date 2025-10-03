from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
import uuid
import posixpath
from datetime import datetime, timezone
import json as _json
from urllib import request as _urlreq

from ..services.s3 import get_s3

router = APIRouter(prefix="/corrections", tags=["corrections"])

CLASS_NAMES = ['overripe', 'ripe', 'rotten', 'unripe', 'unknowns']


def map_days_to_label(days_left: float) -> str:
    """Mappe le nombre de jours restant vers une classe.
    Par défaut:
      days < 0 -> 'rotten'
      0 <= days <= OVERRIPE_MAX -> 'overripe'
      RIPE_MIN <= days <= RIPE_MAX -> 'ripe'
      days >= UNRIPE_MIN -> 'unripe'
      Sinon -> 'ripe' (fallback)
    Variables d'env:
      DAYS_UNRIPE_MIN (def=5), DAYS_RIPE_MIN (def=2), DAYS_RIPE_MAX (def=4), DAYS_OVERRIPE_MAX (def=1)
    """
    unripe_min = float(os.getenv("DAYS_UNRIPE_MIN", "5"))
    ripe_min = float(os.getenv("DAYS_RIPE_MIN", "2"))
    ripe_max = float(os.getenv("DAYS_RIPE_MAX", "4"))
    overripe_max = float(os.getenv("DAYS_OVERRIPE_MAX", "1"))

    if days_left < 0:
        return 'rotten'
    if days_left <= overripe_max:
        return 'overripe'
    if ripe_min <= days_left <= ripe_max:
        return 'ripe'
    if days_left >= unripe_min:
        return 'unripe'
    # gap fallback (e.g., 1 < days < 2 or 4 < days < 5):
    return 'ripe'


class CorrectionRequest(BaseModel):
    image_key: str = Field(..., description="S3 key de l'image uploadée")
    # Option 1: utilisateur fournit directement la classe
    corrected_label: Optional[str] = Field(None, description="Label corrigé parmi classes supportées")
    # Option 2: l'utilisateur dit si c'est une banane et le nombre de jours restants
    is_banana: Optional[bool] = Field(None, description="True si c'est une banane, False sinon")
    days_left: Optional[float] = Field(None, description="Nombre de jours restants si c'est une banane")

    # infos de prédiction pour audit/UX
    predicted_label: Optional[str] = None
    predicted_index: Optional[int] = None
    confidence: Optional[float] = None

    metadata: Optional[Dict[str, Any]] = None  # libre: userId, sessionId, etc.


@router.post("")
async def submit_correction(body: CorrectionRequest) -> Dict[str, Any]:
    # Déterminer la classe finale
    final_label: Optional[str] = None

    if body.corrected_label:
        if body.corrected_label not in CLASS_NAMES:
            raise HTTPException(status_code=400, detail=f"corrected_label must be one of {CLASS_NAMES}")
        final_label = body.corrected_label
    else:
        if body.is_banana is None:
            raise HTTPException(status_code=400, detail="Provide either corrected_label or is_banana (+ days_left if True)")
        if body.is_banana is False:
            final_label = 'unknowns'
        else:
            if body.days_left is None:
                raise HTTPException(status_code=400, detail="days_left is required when is_banana is True")
            final_label = map_days_to_label(float(body.days_left))

    s3 = get_s3()

    dataset_prefix = os.getenv("DATASET_PREFIX", "dataset_new")
    dataset_prefix = s3.ensure_prefix(dataset_prefix)
    corrections_prefix = os.getenv("CORRECTIONS_PREFIX", "corrections")
    corrections_prefix = s3.ensure_prefix(corrections_prefix)
    counter_key = os.getenv("CORRECTION_COUNTER_KEY", "metrics/corrections.json")
    threshold = int(os.getenv("CORRECTION_THRESHOLD", "1000"))

    # générer un id et une destination structurée
    cid = uuid.uuid4().hex
    # extension depuis image_key
    _, ext = os.path.splitext(body.image_key)
    ext = ext if ext else ".jpg"
    dest_key = posixpath.join(dataset_prefix, final_label, f"{cid}{ext}")

    try:
        # copie de l'image dans le dataset structuré
        s3.copy_object(source_key=body.image_key, dest_key=dest_key)

        # record JSON
        record = {
            "id": cid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_key": body.image_key,
            "dest_key": dest_key,
            "final_label": final_label,
            "provided_corrected_label": body.corrected_label,
            "is_banana": body.is_banana,
            "days_left": body.days_left,
            "predicted_label": body.predicted_label,
            "predicted_index": body.predicted_index,
            "confidence": body.confidence,
            "metadata": body.metadata or {},
        }
        record_key = posixpath.join(corrections_prefix, "records", f"{cid}.json")
        s3.put_json(record_key, record)

        # incrément compteur
        count = s3.increment_counter(counter_key)
        threshold_reached = count >= threshold

        # Optionnel: webhook d’alerte
        if threshold_reached:
            webhook = os.getenv("ALERT_WEBHOOK_URL")
            if webhook:
                payload = {
                    "event": "corrections_threshold_reached",
                    "count": count,
                    "threshold": threshold,
                    "latest_record_key": record_key,
                    "latest_dest_key": dest_key,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    req = _urlreq.Request(webhook, data=_json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
                    _urlreq.urlopen(req, timeout=5)
                except Exception:
                    # Ne bloque pas la réponse si l’alerte échoue
                    pass

        return {
            "ok": True,
            "record_key": record_key,
            "dest_key": dest_key,
            "label": final_label,
            "count": count,
            "threshold": threshold,
            "threshold_reached": threshold_reached,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def corrections_stats() -> Dict[str, Any]:
    s3 = get_s3()
    counter_key = os.getenv("CORRECTION_COUNTER_KEY", "metrics/corrections.json")
    data = s3.get_json(counter_key) or {"count": 0}
    return {"count": int(data.get("count", 0)), "updated_at": data.get("updated_at")}
