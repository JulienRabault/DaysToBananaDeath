# Banana Prediction Frontend

Application web React élégante pour la prédiction de maturité des bananes, connectée à un backend FastAPI.

## Fonctionnalités

- 📤 Upload d'images par glisser-déposer
- 📷 Capture photo via webcam/mobile
- 🎯 Prédiction de maturité avec scores de confiance
- ✏️ Système de correction des prédictions
- ⚙️ Configuration flexible des endpoints
- 🌓 Thème clair/sombre/système
- ♿ Entièrement accessible (ARIA, navigation clavier)
- 📱 Design responsive

## Installation

```bash
cd frontend
npm install
```

## Configuration

Créez un fichier `.env` à la racine du dossier `frontend` :

```env
VITE_API_BASE_URL=https://your-backend.up.railway.app
```

Pour le développement local :

```env
VITE_API_BASE_URL=http://localhost:8000
```

## Développement

```bash
npm run dev
```

L'application sera disponible sur `http://localhost:3000`.

## Build

```bash
npm run build
```

Les fichiers de production seront générés dans le dossier `dist/`.

## Tests

```bash
npm run test
```

## Endpoints API

L'application se connecte aux endpoints suivants (configurables via l'interface) :

### Health Check
```bash
GET /health
```

Vérifie l'état du backend.

### Prédiction
```bash
POST /predict
Content-Type: multipart/form-data

{
  "file": <image>
}
```

Réponse attendue :
```json
{
  "predictions": [
    { "label": "ripe", "confidence": 0.87 },
    { "label": "overripe", "confidence": 0.10 }
  ],
  "predicted_label": "ripe",
  "predicted_index": 1,
  "confidence": 0.87,
  "image_key": "uploads/2025/10/06/image.jpg",
  "latency_ms": 123
}
```

### Correction
```bash
POST /corrections
Content-Type: application/json

{
  "image_key": "uploads/2025/10/06/image.jpg",
  "is_banana": true,
  "days_left": 3,
  "predicted_label": "ripe",
  "predicted_index": 1,
  "confidence": 0.87,
  "metadata": {
    "client": "web",
    "ts": "2025-10-06T12:00:00Z"
  }
}
```

Si `is_banana` est `false`, le champ `days_left` n'est pas envoyé.

## Déploiement sur Railway

### Méthode 1 : Build statique (recommandé)

1. Créez un nouveau projet sur Railway
2. Connectez votre repository GitHub
3. Railway détectera automatiquement le projet React/Vite
4. Ajoutez la variable d'environnement :
   ```
   VITE_API_BASE_URL=https://your-backend.up.railway.app
   ```
5. Railway construira et déploiera automatiquement

### Méthode 2 : Nixpacks

Railway utilisera Nixpacks pour détecter et construire votre application automatiquement.

Le `package.json` contient déjà les scripts nécessaires :
- `build` : Construction de production
- `preview` : Serveur de prévisualisation

### Configuration CORS côté backend

Le backend FastAPI doit autoriser les requêtes CORS depuis votre domaine frontend :

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.up.railway.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Pour le développement :
```python
allow_origins=["http://localhost:3000"]
```

## Structure du projet

```
frontend/
├── src/
│   ├── api/              # Client API et endpoints
│   │   ├── client.ts     # Wrapper fetch avec retry et timeout
│   │   └── endpoints.ts  # Fonctions d'appel API
│   ├── components/       # Composants réutilisables
│   │   ├── CameraCapture.tsx
│   │   ├── CorrectionForm.tsx
│   │   ├── ErrorAlert.tsx
│   │   ├── FileDropzone.tsx
│   │   ├── LatencyBadge.tsx
│   │   ├── Layout.tsx
│   │   ├── PredictionResult.tsx
│   │   └── Spinner.tsx
│   ├── config/           # Configuration
│   │   └── api.ts        # Constantes et mapping
│   ├── pages/            # Pages de l'application
│   │   ├── Home.tsx
│   │   ├── Predict.tsx
│   │   └── Settings.tsx
│   ├── store/            # State management (Zustand)
│   │   └── useSettings.ts
│   ├── types/            # Types TypeScript
│   │   └── index.ts
│   ├── utils/            # Utilitaires
│   │   └── theme.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── index.html
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
└── package.json
```

## Personnalisation

### Endpoints
Vous pouvez modifier les endpoints par défaut dans `src/config/api.ts`.

### Mapping des réponses
Si votre backend utilise des clés JSON différentes, configurez-les dans l'interface Settings ou modifiez `DEFAULT_RESPONSE_MAPPING` dans `src/config/api.ts`.

### Thème
Le thème est configurable via l'interface utilisateur (icône en haut à droite) :
- ☀️ Clair
- 🌙 Sombre
- 💻 Système (détecte automatiquement)

## Technologies

- **React 18** - Framework UI
- **TypeScript** - Typage statique
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Zustand** - State management
- **React Router** - Routing
- **React Dropzone** - Upload de fichiers
- **Vitest** - Tests

## Accessibilité

L'application respecte les standards WCAG 2.1 niveau AA :
- Navigation complète au clavier
- Attributs ARIA appropriés
- Contrastes de couleurs suffisants
- Focus visibles
- Textes alternatifs pour les images

## Support navigateurs

- Chrome/Edge (dernières versions)
- Firefox (dernières versions)
- Safari (dernières versions)
- Mobile Safari
- Chrome Android

## Licence

MIT
