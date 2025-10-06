import { useState } from 'react';
import { PredictResponse, CorrectionPayload } from '../types';
import { submitCorrection } from '../api/endpoints';
import { ErrorAlert } from './ErrorAlert';
import { Spinner } from './Spinner';
import { useSettings } from '../store/useSettings';
import { useTranslation, translateClass } from '../utils/i18n';

interface CorrectionFormProps {
  prediction: PredictResponse;
  onSuccess: () => void;
  onReset: () => void;
}

export const CorrectionForm = ({ prediction, onSuccess, onReset }: CorrectionFormProps) => {
  const { language } = useSettings();
  const t = useTranslation(language);

  const [isBanana, setIsBanana] = useState(true);
  const [daysLeft, setDaysLeft] = useState(3);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [correctionSubmitted, setCorrectionSubmitted] = useState(false); // Nouveau state

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
      setCorrectionSubmitted(true);
      // Ne plus appeler onSuccess() - on garde tout affiché
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erreur lors de l\'envoi de la correction');
    } finally {
      setIsSubmitting(false);
    }
  };

  const getPredictedClass = () => {
    if (daysLeft >= 5) return 'unripe';
    if (daysLeft >= 2 && daysLeft <= 4) return 'ripe';
    if (daysLeft === 1) return 'overripe';
    if (daysLeft === 0) return 'rotten';
    return 'unknowns';
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
                {success ? '✅ Correction envoyée' : t.correctionTitle}
              </h3>
              {success && (
                <p className="text-sm text-green-600 dark:text-green-300">
                  Faites une nouvelle prédiction pour corriger à nouveau
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

          {/* Message de succès affiché dans le formulaire */}
          {success && (
            <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg dark:bg-green-900/20 dark:border-green-700">
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5 text-green-600 dark:text-green-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                <span className="text-sm font-medium text-green-800 dark:text-green-200">
                  Correction envoyée avec succès ! Pour soumettre une nouvelle correction, effectuez d'abord une nouvelle prédiction.
                </span>
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Question banane ou pas */}
            <div>
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 block">
                Est-ce bien une banane ?
              </label>
              <div className="flex gap-4">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    checked={isBanana}
                    onChange={() => setIsBanana(true)}
                    disabled={success}
                    className="text-orange-600 focus:ring-orange-500 disabled:opacity-50"
                  />
                  <span className={`text-sm text-gray-700 dark:text-gray-300 ${success ? 'opacity-50' : ''}`}>
                    🍌 {t.yes}
                  </span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    checked={!isBanana}
                    onChange={() => setIsBanana(false)}
                    disabled={success}
                    className="text-orange-600 focus:ring-orange-500 disabled:opacity-50"
                  />
                  <span className={`text-sm text-gray-700 dark:text-gray-300 ${success ? 'opacity-50' : ''}`}>
                    ❌ {t.no}
                  </span>
                </label>
              </div>
            </div>

            {/* Slider pour les jours si c'est une banane */}
            {isBanana && (
              <div>
                <label className={`text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 block ${success ? 'opacity-50' : ''}`}>
                  Jours restants avant que la banane soit trop mûre :
                </label>
                <div className="space-y-4">
                  <div className="flex items-center gap-4">
                    <input
                      type="range"
                      min="0"
                      max="7"
                      value={daysLeft}
                      onChange={(e) => setDaysLeft(parseInt(e.target.value))}
                      disabled={success}
                      className={`flex-1 h-2 bg-gradient-to-r from-red-200 via-yellow-200 to-green-200 rounded-lg appearance-none cursor-pointer dark:from-red-900 dark:via-yellow-900 dark:to-green-900 ${success ? 'opacity-50' : ''}`}
                    />
                    <span className={`text-lg font-semibold text-gray-900 dark:text-white min-w-[3rem] text-center ${success ? 'opacity-50' : ''}`}>
                      {daysLeft}
                    </span>
                  </div>

                  {/* Aperçu de la classification */}
                  <div className={`flex items-center gap-3 p-3 bg-gray-50 rounded-lg dark:bg-gray-700/50 ${success ? 'opacity-50' : ''}`}>
                    <span className="text-lg">{getClassEmoji(getPredictedClass())}</span>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      Cela correspond à : <strong>{translateClass(getPredictedClass(), language)}</strong>
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Bouton de soumission uniquement */}
            {!success && (
              <div className="pt-4">
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-orange-500 to-red-600 px-6 py-3 font-semibold text-white shadow-lg hover:from-orange-600 hover:to-red-700 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:ring-offset-2 disabled:opacity-50 transition-all"
                >
                  {isSubmitting ? (
                    <>
                      <Spinner size="sm" />
                      Envoi...
                    </>
                  ) : (
                    <>
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
                      </svg>
                      {t.correctionSubmit}
                    </>
                  )}
                </button>
              </div>
            )}
          </form>
        </div>
      )}
    </div>
  );
}
