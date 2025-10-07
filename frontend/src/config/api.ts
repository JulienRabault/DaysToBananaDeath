// Configuration de base URL pour l'API
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const DEFAULT_ENDPOINTS = {
  HEALTH: '/health',
  PREDICT: '/api/predict',
  CORRECTIONS: '/api/corrections',
};

export const DEFAULT_RESPONSE_MAPPING = {
  image_key: 'image_key',
  predicted_label: 'predicted_label',
  predicted_index: 'predicted_index',
  confidence: 'confidence',
  predictions: 'predictions',
  latency_ms: 'latency_ms',
};

export const API_CONFIG = {
  TIMEOUT: 30000,
  MAX_RETRIES: 2,
  RETRY_DELAY: 1000,
};
