import { ApiError } from '../types';

interface ErrorAlertProps {
  error: ApiError | Error | string;
  onDismiss?: () => void;
}

export const ErrorAlert = ({ error, onDismiss }: ErrorAlertProps) => {
  const message =
    typeof error === 'string' ? error : 'message' in error ? error.message : String(error);

  const details =
    typeof error === 'object' && 'details' in error ? (error as ApiError).details : undefined;

  return (
    <div
      role="alert"
      className="rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-800 dark:bg-red-950"
    >
      <div className="flex items-start gap-3">
        <svg
          className="h-5 w-5 flex-shrink-0 text-red-600 dark:text-red-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        <div className="flex-1">
          <p className="text-sm font-medium text-red-800 dark:text-red-200">{message}</p>
          {details && (
            <p className="mt-1 text-xs text-red-700 dark:text-red-300">{details}</p>
          )}
        </div>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="flex-shrink-0 rounded p-1 hover:bg-red-100 focus:outline-none focus:ring-2 focus:ring-red-500 dark:hover:bg-red-900"
            aria-label="Fermer l'alerte"
          >
            <svg className="h-4 w-4 text-red-600 dark:text-red-400" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
};
