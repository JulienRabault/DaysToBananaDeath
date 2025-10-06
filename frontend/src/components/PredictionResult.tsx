import { PredictResponse } from '../types';
import { LatencyBadge } from './LatencyBadge';

interface PredictionResultProps {
  prediction: PredictResponse;
}

export const PredictionResult = ({ prediction }: PredictionResultProps) => {
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
    if (confidence >= 0.5) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
    return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
  };

  return (
    <div className="space-y-6 rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Résultat de la prédiction
          </h3>
          <div className="mt-2 flex items-center gap-3">
            <span
              className={`rounded-full px-4 py-1.5 text-sm font-medium ${getConfidenceColor(
                prediction.confidence
              )}`}
            >
              {prediction.predicted_label}
            </span>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {(prediction.confidence * 100).toFixed(1)}% de confiance
            </span>
          </div>
        </div>
        <LatencyBadge latency={prediction.latency_ms} />
      </div>

      <div>
        <h4 className="mb-3 text-sm font-medium text-gray-700 dark:text-gray-300">
          Top prédictions
        </h4>
        <div className="space-y-2">
          {prediction.predictions.map((pred, index) => (
            <div key={index} className="flex items-center gap-3">
              <span className="w-24 text-sm text-gray-600 dark:text-gray-400">{pred.label}</span>
              <div className="flex-1">
                <div className="h-2 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
                  <div
                    className="h-full rounded-full bg-primary-600 transition-all duration-300 dark:bg-primary-500"
                    style={{ width: `${pred.confidence * 100}%` }}
                    role="progressbar"
                    aria-valuenow={pred.confidence * 100}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-label={`Confiance pour ${pred.label}`}
                  />
                </div>
              </div>
              <span className="w-12 text-right text-sm font-medium text-gray-900 dark:text-gray-100">
                {(pred.confidence * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {prediction.image_key && (
        <div className="border-t border-gray-200 pt-4 dark:border-gray-700">
          <p className="text-xs text-gray-500 dark:text-gray-400">
            <span className="font-medium">Clé d&apos;image :</span> {prediction.image_key}
          </p>
        </div>
      )}
    </div>
  );
};
