export interface Prediction {
  label: string;
  confidence: number;
}

export interface PredictResponse {
  predictions: Prediction[];
  predicted_label: string;
  predicted_index: number;
  confidence: number;
  image_key: string;
  latency_ms: number;
  // For temporary images that aren't uploaded to S3 yet
  temp_file_data?: {
    content: string;
    filename: string;
    content_type: string;
  };
}

export interface CorrectionPayload {
  image_key: string;
  // Include temp file data for images not yet uploaded to S3
  temp_file_data?: {
    content: string;
    filename: string;
    content_type: string;
  };
  is_banana: boolean;
  days_left?: number;
  predicted_label: string;
  predicted_index: number;
  confidence: number;
  metadata: {
    client: string;
    ts: string;
  };
}

export interface HealthResponse {
  status: string;
  timestamp?: string;
}

export interface ApiError {
  message: string;
  status?: number;
  details?: string;
}
