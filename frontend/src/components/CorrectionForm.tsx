import { useState } from 'react';
import { PredictResponse, CorrectionPayload } from '../types';
import { submitCorrection } from '../api/endpoints';
import { ErrorAlert } from './ErrorAlert';
import { Spinner } from './Spinner';
import { useSettings } from '../store/useSettings';
import { useTranslation, translateClass } from '../utils/i18n';

interface CorrectionFormProps {
  prediction: PredictResponse;
  // onSuccess: () => void;
  // onReset: () => void;
}

export const CorrectionForm = ({ prediction }: CorrectionFormProps) => {
  const { language } = useSettings();
  const t = useTranslation(language);

  const [isBanana, setIsBanana] = useState(true);
  const [daysLeft, setDaysLeft] = useState(3);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  // const [correctionSubmitted, setCorrectionSubmitted] = useState(false); // Variable non utilisée

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsSubmitting(true);

    try {
      const payload: CorrectionPayload = {
        image_key: prediction.image_key,
        is_banana: isBanana,
        ...(isBanana && { days_left: daysLeft }),
        predicted_label: prediction.predicted_label,
        predicted_index: prediction.predicted_index,
        confidence: prediction.confidence,
        metadata: {
          client: 'web',
          ts: new Date().toISOString(),
        },
      };

      if (prediction.temp_file_data) {
        payload.temp_file_data = prediction.temp_file_data;
      }

      await submitCorrection(payload);
      setSuccess(true);
      // Ne plus appeler onSuccess() - on garde tout affiché
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erreur lors de l\'envoi de la correction');
    } finally {
      setIsSubmitting(false);
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
    <div className="rounded-2xl border border-yellow-200/50 bg-white/60 backdrop-blur-sm shadow-lg dark:border-gray-700/50 dark:bg-gray-800/60">
      {/* Header avec bouton d'expansion et bouton reset intégré */}
      <div className="p-6 border-b border-gray-200/50 dark:border-gray-700/50">
        <div className="flex items-center justify-between">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-3 text-left flex-1"
            aria-expanded={isExpanded}
          >
            <div className={`p-2 rounded-lg ${success ? 'bg-green-100 dark:bg-green-900/30' : 'bg-orange-100 dark:bg-orange-900/30'}`}>
              {success ? (
                <svg className="w-5 h-5 text-green-600 dark:text-green-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              ) : (
                <svg className="w-5 h-5 text-orange-600 dark:text-orange-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
                </svg>
                )}
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                {success ? t.correctionSuccessTitle : t.correctionTitle}
              </h3>
              {success && (
                <p className="text-sm text-green-600 dark:text-green-300">
                  {t.correctionNewPredictionMessage}
                </p>
              )}
            </div>
          </button>

          {/* Boutons d'action dans l'en-tête */}
          <div className="flex items-center gap-3">
            <svg
              className={`w-5 h-5 text-gray-500 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </div>
        </div>
      </div>

      {/* Contenu du formulaire */}
      {isExpanded && (
        <div className="p-6">
          {error && (
            <div className="mb-6">
              <ErrorAlert error={error} onDismiss={() => setError(null)} />
            </div>
          )}

          {/* Message de succès moderne */}
          {success && (
            <div className="mb-6 p-4 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200/50 rounded-xl shadow-sm dark:from-green-900/20 dark:to-emerald-900/20 dark:border-green-700/50">
              <div className="flex items-center gap-3">
                <div className="flex-shrink-0 w-8 h-8 bg-green-100 rounded-full flex items-center justify-center dark:bg-green-900/40">
                  <svg className="w-5 h-5 text-green-600 dark:text-green-400" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-green-800 dark:text-green-200">
                    {t.correctionSuccessTitle}
                  </h4>
                  <p className="text-xs text-green-600 dark:text-green-300 mt-1">
                    {t.correctionSuccessMessage}
                  </p>
                </div>
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Question banane ou pas - Design simplifié */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                {t.correctionBananaQuestion}
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <label className={`
                  relative flex items-center gap-3 p-4 rounded-lg border cursor-pointer transition-all
                  ${isBanana 
                    ? 'border-green-500 bg-green-50 dark:border-green-400 dark:bg-green-900/20' 
                    : 'border-gray-300 hover:border-gray-400 dark:border-gray-600 dark:hover:border-gray-500'
                  }
                  ${success ? 'opacity-50 cursor-not-allowed' : ''}
                `}>
                  <input
                    type="radio"
                    checked={isBanana}
                    onChange={() => setIsBanana(true)}
                    disabled={success}
                    className="w-4 h-4 text-green-600 border-gray-300 focus:ring-green-500"
                  />
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {t.correctionYesBanana}
                  </span>
                </label>

                <label className={`
                  relative flex items-center gap-3 p-4 rounded-lg border cursor-pointer transition-all
                  ${!isBanana 
                    ? 'border-red-500 bg-red-50 dark:border-red-400 dark:bg-red-900/20' 
                    : 'border-gray-300 hover:border-gray-400 dark:border-gray-600 dark:hover:border-gray-500'
                  }
                  ${success ? 'opacity-50 cursor-not-allowed' : ''}
                `}>
                  <input
                    type="radio"
                    checked={!isBanana}
                    onChange={() => setIsBanana(false)}
                    disabled={success}
                    className="w-4 h-4 text-red-600 border-gray-300 focus:ring-red-500"
                  />
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {t.correctionNoBanana}
                  </span>
                </label>
              </div>
            </div>

            {/* Slider moderne pour les jours si c'est une banane */}
            {isBanana && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                    {t.correctionDurationTitle}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {t.correctionDurationSubtitle}
                  </p>
                </div>

                <div className="bg-gray-50 dark:bg-gray-800/30 rounded-xl p-6 border border-gray-200/50 dark:border-gray-700/50">
                  <div className="flex items-center gap-6">
                    <div className="flex-1">
                      <input
                        type="range"
                        min="0"
                        max="7"
                        value={daysLeft}
                        onChange={(e) => setDaysLeft(parseInt(e.target.value))}
                        disabled={success}
                        className={`w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700 slider ${success ? 'opacity-50' : ''}`}
                        style={{
                          background: `linear-gradient(to right, 
                            #ef4444 0%, 
                            #f59e0b 30%, 
                            #10b981 60%, 
                            #059669 100%)`
                        }}
                      />
                    </div>
                    <div className={`text-center min-w-[80px] ${success ? 'opacity-50' : ''}`}>
                      <div className="text-2xl font-bold text-gray-900 dark:text-white">
                        {daysLeft}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                        {daysLeft > 1 ? t.correctionDays : t.correctionDay}
                      </div>
                    </div>
                  </div>

                  {/* Échelle avec indicateurs discrets */}
                  <div className="mt-4 flex justify-between items-center text-xs">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-red-400"></div>
                      <span className="text-gray-600 dark:text-gray-400 font-medium">{t.correctionStateImmediate}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
                      <span className="text-gray-600 dark:text-gray-400 font-medium">{t.correctionStateOptimal}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-green-400"></div>
                      <span className="text-gray-600 dark:text-gray-400 font-medium">{t.correctionStateConservation}</span>
                    </div>
                  </div>

                  {/* Indication contextuelle basée sur la valeur */}
                  <div className="mt-4 p-3 rounded-lg bg-white dark:bg-gray-900/50 border border-gray-200/50 dark:border-gray-700/50">
                    <div className="text-sm">
                      <span className="font-medium text-gray-900 dark:text-white">{t.correctionStatePredicted}: </span>
                      <span className={`
                        ${daysLeft === 0 ? 'text-red-600 dark:text-red-400' :
                          daysLeft <= 2 ? 'text-orange-600 dark:text-orange-400' :
                          daysLeft <= 4 ? 'text-yellow-600 dark:text-yellow-400' :
                          'text-green-600 dark:text-green-400'}
                      `}>
                        {daysLeft === 0 ? t.stateVeryRipe :
                         daysLeft === 1 ? t.stateRipe :
                         daysLeft <= 2 ? t.stateGoodRipeness :
                         daysLeft <= 4 ? t.stateMediumRipeness :
                         t.stateStillGreen}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Bouton de soumission moderne */}
            {!success && (
              <div className="pt-4">
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="flex w-full items-center justify-center gap-3 rounded-xl bg-gradient-to-r from-orange-500 to-red-600 px-6 py-4 font-semibold text-white shadow-lg hover:from-orange-600 hover:to-red-700 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
                >
                  {isSubmitting ? (
                    <>
                      <Spinner size="sm" />
                      <span>{t.correctionSending}</span>
                    </>
                  ) : (
                    <>
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-8.293l-3-3a1 1 0 00-1.414 1.414L10.586 9H7a1 1 0 100 2h3.586l-1.293 1.293a1 1 0 101.414 1.414l3-3a1 1 0 000-1.414z" clipRule="evenodd" />
                      </svg>
                      <span>{t.correctionSendButton}</span>
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Informations sur la prédiction actuelle */}
            <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-xl border border-gray-200/50 dark:border-gray-700/50">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2">
                <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                {t.correctionCurrentPrediction}
              </h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-600 dark:text-gray-400">{t.resultTitle}:</span>
                  <div className="font-medium text-gray-900 dark:text-white flex items-center gap-2 mt-1">
                    {getClassEmoji(prediction.predicted_label)}
                    {translateClass(prediction.predicted_label, language)}
                  </div>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">{t.resultConfidence}:</span>
                  <div className="font-medium text-gray-900 dark:text-white mt-1">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          </form>
        </div>
      )}
    </div>
  );
};
