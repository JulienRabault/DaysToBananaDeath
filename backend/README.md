Banana Ripeness Backend (FastAPI)

Endpoints
- GET /health: ping
- POST /api/presign/upload: retourne une URL pré-signée S3 pour uploader une image directement depuis le frontend
  Body: { content_type?: string, folder?: string, filename?: string, use_put?: boolean }
  Return: { method: "PUT"|"POST", url|post, key, bucket }
- POST /api/predict/file: upload multipart d’une image et renvoie la prédiction
- POST /api/predict/s3: prédiction depuis une clé S3 existante
  Body: { key: string }
- POST /api/corrections: enregistre une correction utilisateur et copie l’image dans un dataset structuré
  Body: Option A: { image_key, corrected_label }
        Option B: { image_key, is_banana: false } => class "unknowns"
        Option C: { image_key, is_banana: true, days_left: number }
  Retour: { ok, label, dest_key, record_key, count, threshold, threshold_reached }
- GET /api/corrections/stats: nombre de corrections enregistrées

Classes utilisées: ["overripe", "ripe", "rotten", "unripe", "unknowns"]

Mapping days_left -> label (par défaut)
- days < 0 => rotten
- 0 <= days <= DAYS_OVERRIPE_MAX (def=1) => overripe
- DAYS_RIPE_MIN..DAYS_RIPE_MAX (def=2..4) => ripe
- days >= DAYS_UNRIPE_MIN (def=5) => unripe
- sinon => ripe

Variables d’environnement (Railway)
- S3_BUCKET (requis)
- AWS_REGION (def=eu-west-3)
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- S3_ENDPOINT_URL (optionnel: MinIO/custom)
- FRONTEND_ORIGIN (CORS, ex: https://votre-front.app)
- MODEL_FORMAT: onnx|ckpt (def=ckpt)
- MODEL_S3_KEY: clé S3 vers le modèle (.onnx ou .ckpt)
- MODEL_IMG_SIZE (def=224)
- MODEL_TYPE: resnet50|vit_b_16 (def=resnet50)
- INFERENCE_DEVICE (def=cpu)
- MODEL_TMP_DIR_PARENT (optionnel: dossier parent pour le tempdir modèle)
- UPLOAD_PREFIX (def=incoming)
- DATASET_PREFIX (def=dataset_new)
- CORRECTIONS_PREFIX (def=corrections)
- CORRECTION_COUNTER_KEY (def=metrics/corrections.json)
- CORRECTION_THRESHOLD (def=1000)
- DAYS_UNRIPE_MIN (def=5)
- DAYS_RIPE_MIN (def=2)
- DAYS_RIPE_MAX (def=4)
- DAYS_OVERRIPE_MAX (def=1)
- (optionnel W&B si usage .ckpt par artifact): WANDB_RUN_PATH, WANDB_ARTIFACT, WANDB_API_KEY

Cycle de vie (téléchargement modèle)
- Au démarrage (startup): téléchargement du modèle depuis S3 dans un répertoire temporaire, puis chargement (ONNX Runtime ou PyTorch via ml/src/inference.py)
- À l’extinction (shutdown): suppression du répertoire temporaire

Démarrage local
- Installer les dépendances: pip install -r requirements.txt (racine) ou backend/requirements.txt
- Lancer: uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

Notes
- Si vous utilisez MODEl_FORMAT=ckpt, assurez-vous d’installer les libs PyTorch/Lightning/Albumentations/OMEGACONF (voir backend/requirements.txt pour exemples).
- /api/presign/upload renvoie la clé S3; le frontend doit faire un PUT direct du fichier vers l’URL signée, puis appeler /api/predict/s3 avec la clé pour prédire, puis /api/corrections pour enregistrer le label final.

