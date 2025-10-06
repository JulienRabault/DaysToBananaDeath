import { useState } from 'react';
import { useSettings } from '../store/useSettings';
import { health } from '../api/endpoints';
import { ErrorAlert } from '../components/ErrorAlert';
import { Spinner } from '../components/Spinner';

export const Settings = () => {
  const settings = useSettings();
  const [localBaseUrl, setLocalBaseUrl] = useState(settings.baseUrl);
  const [localEndpoints, setLocalEndpoints] = useState(settings.endpoints);
  const [localMapping, setLocalMapping] = useState(settings.responseMapping);
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState<'success' | 'error' | null>(null);
  const [testError, setTestError] = useState<string | null>(null);

  const handleSave = () => {
    settings.setBaseUrl(localBaseUrl);
    Object.entries(localEndpoints).forEach(([key, value]) => {
      settings.setEndpoint(key as keyof typeof localEndpoints, value);
    });
    Object.entries(localMapping).forEach(([key, value]) => {
      settings.setResponseMapping(key as keyof typeof localMapping, value);
    });
  };

  const handleReset = () => {
    settings.resetToDefaults();
    setLocalBaseUrl(settings.baseUrl);
    setLocalEndpoints(settings.endpoints);
    setLocalMapping(settings.responseMapping);
  };

  const handleTest = async () => {
    setIsTesting(true);
    setTestResult(null);
    setTestError(null);

    const oldSettings = {
      baseUrl: settings.baseUrl,
      endpoints: { ...settings.endpoints },
    };

    try {
      settings.setBaseUrl(localBaseUrl);
      Object.entries(localEndpoints).forEach(([key, value]) => {
        settings.setEndpoint(key as keyof typeof localEndpoints, value);
      });

      await health();
      setTestResult('success');
    } catch (err) {
      setTestResult('error');
      setTestError(err instanceof Error ? err.message : 'Échec de la connexion');

      settings.setBaseUrl(oldSettings.baseUrl);
      Object.entries(oldSettings.endpoints).forEach(([key, value]) => {
        settings.setEndpoint(key as keyof typeof oldSettings.endpoints, value);
      });
    } finally {
      setIsTesting(false);
    }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Paramètres</h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Configurez la connexion au backend et les endpoints de l&apos;API
        </p>
      </div>

      <div className="space-y-6 rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
        <div>
          <h2 className="mb-4 text-lg font-semibold text-gray-900 dark:text-white">
            Configuration du backend
          </h2>
          <div className="space-y-4">
            <div>
              <label htmlFor="base-url" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                URL de base du backend
              </label>
              <input
                type="url"
                id="base-url"
                value={localBaseUrl}
                onChange={(e) => setLocalBaseUrl(e.target.value)}
                placeholder="https://your-backend.up.railway.app"
                className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500 dark:border-gray-600 dark:bg-gray-700"
              />
            </div>
          </div>
        </div>

        <div>
          <h2 className="mb-4 text-lg font-semibold text-gray-900 dark:text-white">
            Endpoints
          </h2>
          <div className="space-y-4">
            {Object.entries(localEndpoints).map(([key, value]) => (
              <div key={key}>
                <label htmlFor={`endpoint-${key}`} className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  {key.charAt(0).toUpperCase() + key.slice(1)}
                </label>
                <input
                  type="text"
                  id={`endpoint-${key}`}
                  value={value}
                  onChange={(e) =>
                    setLocalEndpoints({ ...localEndpoints, [key]: e.target.value })
                  }
                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500 dark:border-gray-600 dark:bg-gray-700"
                />
              </div>
            ))}
          </div>
        </div>

        <div>
          <h2 className="mb-4 text-lg font-semibold text-gray-900 dark:text-white">
            Mapping des réponses
          </h2>
          <p className="mb-4 text-sm text-gray-600 dark:text-gray-400">
            Configurez les clés JSON utilisées dans les réponses du backend
          </p>
          <div className="space-y-4">
            {Object.entries(localMapping).map(([key, value]) => (
              <div key={key}>
                <label htmlFor={`mapping-${key}`} className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  {key}
                </label>
                <input
                  type="text"
                  id={`mapping-${key}`}
                  value={value}
                  onChange={(e) =>
                    setLocalMapping({ ...localMapping, [key]: e.target.value })
                  }
                  className="mt-1 block w-full rounded-lg border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500 dark:border-gray-600 dark:bg-gray-700"
                />
              </div>
            ))}
          </div>
        </div>

        {testResult === 'success' && (
          <div className="rounded-lg border border-green-200 bg-green-50 p-4 dark:border-green-800 dark:bg-green-950">
            <p className="text-sm font-medium text-green-800 dark:text-green-200">
              ✓ Connexion réussie au backend
            </p>
          </div>
        )}

        {testResult === 'error' && testError && (
          <ErrorAlert error={testError} onDismiss={() => setTestResult(null)} />
        )}

        <div className="flex gap-3">
          <button
            onClick={handleTest}
            disabled={isTesting}
            className="flex items-center gap-2 rounded-lg border border-gray-300 px-4 py-2 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50 dark:border-gray-600 dark:hover:bg-gray-700"
          >
            {isTesting ? (
              <>
                <Spinner size="sm" />
                Test en cours...
              </>
            ) : (
              'Tester la connexion'
            )}
          </button>
          <button
            onClick={handleSave}
            className="rounded-lg bg-primary-600 px-4 py-2 text-white hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
          >
            Enregistrer
          </button>
          <button
            onClick={handleReset}
            className="rounded-lg border border-gray-300 px-4 py-2 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 dark:border-gray-600 dark:hover:bg-gray-700"
          >
            Réinitialiser
          </button>
        </div>
      </div>
    </div>
  );
};
