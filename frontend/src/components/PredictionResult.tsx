import { PredictResponse } from '../types';
import { LatencyBadge } from './LatencyBadge';
import { useSettings } from '../store/useSettings';
import { useTranslation, translateClass } from '../utils/i18n';

interface PredictionResultProps {
  prediction: PredictResponse;
}

export const PredictionResult = ({ prediction }: PredictionResultProps) => {
  const { language } = useSettings();
  const t = useTranslation(language);

  const getClassColor = (className: string) => {
    switch (className.toLowerCase()) {
      case 'unripe': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      case 'ripe': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
      case 'overripe': return 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200';
      case 'rotten': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      case 'unknowns':
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'; // Fond gris pour "Inconnue"
    }
  };

  const getClassEmoji = (className: string) => {
    switch (className.toLowerCase()) {
      case 'unripe': return '🟢';
      case 'ripe': return '🟡';
      case 'overripe': return '🟠';
      case 'rotten': return '🟤';
      default: return '❓';
    }
  };

  return (
    <div className="space-y-6 rounded-2xl border border-yellow-200/50 bg-white/60 backdrop-blur-sm p-6 shadow-lg dark:border-gray-700/50 dark:bg-gray-800/60">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              {t.resultTitle}
            </h3>
            <LatencyBadge latency={prediction.latency_ms} />
          </div>
          <div className="flex items-center gap-3">
            <span
              className={`inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold ${getClassColor(
                prediction.predicted_label
              )}`}
            >
              <span className="text-base">{getClassEmoji(prediction.predicted_label)}</span>
              {translateClass(prediction.predicted_label, language)}
            </span>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {(prediction.confidence * 100).toFixed(1)}% {t.resultConfidence}
            </span>
          </div>
        </div>
      </div>

      <div>
        <h4 className="mb-4 flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300">
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M3 3a1 1 0 000 2v8a2 2 0 002 2h2.586l-1.293 1.293a1 1 0 101.414 1.414L10 15.414l2.293 2.293a1 1 0 001.414-1.414L12.414 15H15a2 2 0 002-2V5a1 1 0 100-2H3zm11.707 4.707a1 1 0 00-1.414-1.414L10 9.586 8.707 8.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
          {t.resultTopPredictions}
        </h4>
        <div className="space-y-3">
          {prediction.predictions.map((pred, index) => (
            <div key={index} className="flex items-center gap-4">
              <div className="flex items-center gap-2 w-32">
                <span className="text-sm">{getClassEmoji(pred.label)}</span>
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {translateClass(pred.label, language)}
                </span>
              </div>
              <div className="flex-1">
                <div className="h-2.5 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-yellow-500 to-orange-500 transition-all duration-500 ease-out"
                    style={{ width: `${pred.confidence * 100}%` }}
                    role="progressbar"
                    aria-valuenow={pred.confidence * 100}
                    aria-valuemin={0}
                    aria-valuemax={100}
                  />
                </div>
              </div>
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400 w-12 text-right">
                {(pred.confidence * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {prediction.image_key && (
        <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
          <p className="text-xs text-gray-500 dark:text-gray-400">
            {t.resultImageKey}: <code className="bg-gray-100 dark:bg-gray-700 px-1 py-0.5 rounded text-xs">{prediction.image_key}</code>
          </p>
        </div>
      )}
    </div>
  );
};
