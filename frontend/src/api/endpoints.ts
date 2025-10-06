import { apiClient } from './client';
import { useSettings } from '../store/useSettings';
import { PredictResponse, CorrectionPayload, HealthResponse } from '../types';

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
  const { endpoints, responseMapping } = useSettings.getState();

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
    predictions: (response[responseMapping.predictions] as PredictResponse['predictions']) || [],
    predicted_label: String(response[responseMapping.predicted_label] || ''),
    predicted_index: Number(response[responseMapping.predicted_index] || 0),
    confidence: Number(response[responseMapping.confidence] || 0),
    image_key: String(response[responseMapping.image_key] || ''),
    latency_ms: Number(response[responseMapping.latency_ms] || clientLatency),
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
