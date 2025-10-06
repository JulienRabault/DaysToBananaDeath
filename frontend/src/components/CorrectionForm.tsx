import { useState } from 'react';
import { PredictResponse, CorrectionPayload } from '../types';
import { submitCorrection } from '../api/endpoints';
import { ErrorAlert } from './ErrorAlert';
import { Spinner } from './Spinner';

interface CorrectionFormProps {
  prediction: PredictResponse;
  onSuccess: () => void;
  onReset: () => void; // Nouvelle prop pour le reset
}

export const CorrectionForm = ({ prediction, onSuccess, onReset }: CorrectionFormProps) => {
  const [isBanana, setIsBanana] = useState(true);
  const [daysLeft, setDaysLeft] = useState(3);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false); // État pour le menu dépliable

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

      // Include temp file data if this is a temporary image
      if (prediction.temp_file_data) {
        payload.temp_file_data = prediction.temp_file_data;
      }

      await submitCorrection(payload);
      setSuccess(true);
      setTimeout(() => {
        onSuccess();
      }, 1500);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erreur lors de l\'envoi de la correction');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (success) {
    return (
      <div className="rounded-lg border border-green-200 bg-green-50 p-6 text-center dark:border-green-800 dark:bg-green-950">
        <svg
          className="mx-auto h-12 w-12 text-green-600 dark:text-green-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M5 13l4 4L19 7"
          />
        </svg>
        <p className="mt-2 font-medium text-green-800 dark:text-green-200">
          Correction envoyée avec succès
        </p>
        <button
          onClick={onReset}
          className="mt-3 inline-flex items-center gap-2 rounded-lg bg-green-600 px-4 py-2 text-sm font-medium text-white hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 dark:bg-green-700 dark:hover:bg-green-600"
        >
          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Nouvelle prédiction
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-4 rounded-lg border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800">
      {/* Header avec boutons */}
      <div className="flex items-center justify-between gap-3 p-4 pb-0">
        <button
          onClick={onReset}
          className="inline-flex items-center gap-2 rounded-lg bg-gray-100 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600"
        >
          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Nouvelle prédiction
        </button>

        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="inline-flex items-center gap-2 rounded-lg bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
        >
          <svg
            className={`h-4 w-4 transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
          {isExpanded ? 'Masquer' : 'Soumettre une correction'}
        </button>
      </div>

      {/* Contenu dépliable */}
      <div className={`overflow-hidden transition-all duration-300 ease-in-out ${isExpanded ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'}`}>
        <form onSubmit={handleSubmit} className="space-y-6 p-4 pt-0">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Corriger la prédiction</h3>

          {error && <ErrorAlert error={error} onDismiss={() => setError(null)} />}

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label htmlFor="is-banana" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                C&apos;est une banane ?
              </label>
              <button
                type="button"
                role="switch"
                aria-checked={isBanana}
                id="is-banana"
                onClick={() => setIsBanana(!isBanana)}
                className={`
                  relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent
                  transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2
                  ${isBanana ? 'bg-primary-600' : 'bg-gray-200 dark:bg-gray-700'}
                `}
              >
                <span
                  className={`
                    pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0
                    transition duration-200 ease-in-out
                    ${isBanana ? 'translate-x-5' : 'translate-x-0'}
                  `}
                />
              </button>
            </div>

            {isBanana && (
              <div className="space-y-2">
                <label htmlFor="days-left" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Jours restants avant péremption
                </label>
                <div className="flex items-center gap-4">
                  <input
                    type="range"
                    id="days-left"
                    min="0"
                    max="10"
                    step="0.5"
                    value={daysLeft}
                    onChange={(e) => setDaysLeft(parseFloat(e.target.value))}
                    className="flex-1 accent-primary-600"
                    aria-valuemin={0}
                    aria-valuemax={10}
                    aria-valuenow={daysLeft}
                  />
                  <input
                    type="number"
                    min="0"
                    max="10"
                    step="0.5"
                    value={daysLeft}
                    onChange={(e) => setDaysLeft(parseFloat(e.target.value))}
                    className="w-20 rounded-lg border border-gray-300 px-3 py-1.5 text-center focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500 dark:border-gray-600 dark:bg-gray-700"
                    aria-label="Nombre de jours"
                  />
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {daysLeft === 0 ? 'Périmée' : daysLeft === 1 ? '1 jour' : `${daysLeft} jours`}
                </p>
              </div>
            )}
          </div>

          <button
            type="submit"
            disabled={isSubmitting}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary-600 px-4 py-2.5 font-medium text-white hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50"
          >
            {isSubmitting ? (
              <>
                <Spinner size="sm" />
                Envoi en cours...
              </>
            ) : (
              'Envoyer la correction'
            )}
          </button>
        </form>
      </div>
    </div>
  );
};
