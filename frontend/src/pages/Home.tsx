import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { health } from '../api/endpoints';
import { ErrorAlert } from '../components/ErrorAlert';
import { Spinner } from '../components/Spinner';

export const Home = () => {
  const [status, setStatus] = useState<'loading' | 'online' | 'offline'>('loading');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await health();
        setStatus('online');
      } catch (err) {
        setStatus('offline');
        setError(err instanceof Error ? err.message : 'Backend non disponible');
      }
    };

    checkHealth();
  }, []);

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
          Bienvenue sur Banana Prediction
        </h1>
        <p className="mt-4 text-lg text-gray-600 dark:text-gray-400">
          Analysez vos images de bananes et obtenez des prédictions de maturité
        </p>
      </div>

      <div className="mx-auto max-w-md">
        <div className="rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
          <h2 className="mb-4 text-lg font-semibold text-gray-900 dark:text-white">
            État du backend
          </h2>

          {status === 'loading' && (
            <div className="flex items-center gap-3">
              <Spinner size="sm" />
              <span className="text-gray-600 dark:text-gray-400">Vérification en cours...</span>
            </div>
          )}

          {status === 'online' && (
            <div className="flex items-center gap-3">
              <div className="h-3 w-3 rounded-full bg-green-500" aria-hidden="true" />
              <span className="font-medium text-green-700 dark:text-green-400">En ligne</span>
            </div>
          )}

          {status === 'offline' && (
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="h-3 w-3 rounded-full bg-red-500" aria-hidden="true" />
                <span className="font-medium text-red-700 dark:text-red-400">Hors ligne</span>
              </div>
              {error && <ErrorAlert error={error} />}
            </div>
          )}
        </div>
      </div>

      <div className="mx-auto grid max-w-4xl gap-6 sm:grid-cols-2">
        <Link
          to="/predict"
          className="group rounded-lg border border-gray-200 bg-white p-6 transition-shadow hover:shadow-lg dark:border-gray-700 dark:bg-gray-800"
        >
          <div className="mb-4 text-4xl" aria-hidden="true">🔍</div>
          <h3 className="mb-2 text-xl font-semibold text-gray-900 dark:text-white">
            Faire une prédiction
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            Uploadez une image ou prenez une photo pour obtenir une prédiction de maturité
          </p>
        </Link>

        <Link
          to="/settings"
          className="group rounded-lg border border-gray-200 bg-white p-6 transition-shadow hover:shadow-lg dark:border-gray-700 dark:bg-gray-800"
        >
          <div className="mb-4 text-4xl" aria-hidden="true">⚙️</div>
          <h3 className="mb-2 text-xl font-semibold text-gray-900 dark:text-white">
            Paramètres
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            Configurez l&apos;URL du backend et les endpoints de l&apos;API
          </p>
        </Link>
      </div>

      <div className="mx-auto max-w-2xl rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
        <h2 className="mb-4 text-xl font-semibold text-gray-900 dark:text-white">
          Documentation
        </h2>
        <div className="space-y-3 text-sm text-gray-600 dark:text-gray-400">
          <p>
            Cette application permet d&apos;interagir avec un backend FastAPI pour la prédiction de maturité des bananes.
          </p>
          <ul className="list-inside list-disc space-y-1">
            <li>Uploadez une image depuis votre appareil ou prenez une photo</li>
            <li>Obtenez une prédiction avec un score de confiance</li>
            <li>Soumettez des corrections pour améliorer le modèle</li>
            <li>Configurez les paramètres de connexion au backend</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
