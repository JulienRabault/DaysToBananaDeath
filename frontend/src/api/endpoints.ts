import { apiClient } from './client';
import { useSettings } from '../store/useSettings';
import { PredictResponse, CorrectionPayload, HealthResponse } from '../types';

// Define the model info interface
export interface ModelInfo {
  name: string;
  version: string;
  format: string;
  imageSize: string;
  modelType: string;
  device: string;
  classes: string[];
  classMappings?: {
    days: Record<string, number>;
    thresholds: Record<string, number>;
  };
  modelSources?: {
    localPath?: string;
    s3Key?: string;
    wandbPath?: string;
    wandbArtifact?: string;
  };
  configuration?: {
    mockPredictions: boolean;
    logLevel: string;
    s3Bucket?: string;
    awsRegion: string;
  };
  modelLoaded: boolean;
  canLoadModel: boolean;
  modelMetadata?: Record<string, any>;
  loadedAt?: number;
  error?: string;
}

const getFullUrl = (endpoint: string): string => {
  const { baseUrl } = useSettings.getState();
  const cleanBase = baseUrl.replace(/\/$/, '');
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  return `${cleanBase}${cleanEndpoint}`;
};

export const health = async (signal?: AbortSignal): Promise<HealthResponse> => {
  const { endpoints } = useSettings.getState();
  return apiClient.get<HealthResponse>(getFullUrl(endpoints.health), signal);
};

export const predictImage = async (file: File, signal?: AbortSignal): Promise<PredictResponse> => {
  const { endpoints } = useSettings.getState();

  const formData = new FormData();
  formData.append('file', file);

  const startTime = Date.now();
  const response = await apiClient.postFormData<Record<string, unknown>>(
    getFullUrl(endpoints.predict),
    formData,
    signal
  );
  const clientLatency = Date.now() - startTime;

  const mappedResponse: PredictResponse = {
    predictions: Object.entries(response.class_probabilities || {}).map(([label, confidence]) => ({
      label,
      confidence: Number(confidence)
    })).sort((a, b) => b.confidence - a.confidence),
    predicted_label: String(response.predicted_class || ''),
    predicted_index: Number(response.predicted_index || 0),
    confidence: Number(response.confidence || 0),
    image_key: String(response.image_key || ''),
    latency_ms: Number(response.latency_ms || clientLatency),
    // Include temp_file_data if present in response
    temp_file_data: response.temp_file_data as PredictResponse['temp_file_data'],
  };

  return mappedResponse;
};

export const submitCorrection = async (
  payload: CorrectionPayload,
  signal?: AbortSignal
): Promise<void> => {
  const { endpoints } = useSettings.getState();
  await apiClient.post<void>(getFullUrl(endpoints.corrections), payload, signal);
};

export const getModelInfo = async (signal?: AbortSignal): Promise<ModelInfo> => {
  return apiClient.get<ModelInfo>(getFullUrl('/api/model/info'), signal);
};
