import { useState, useEffect } from 'react';
import { health, predictImage, getModelInfo, type ModelInfo } from '../api/endpoints';
import { PredictResponse } from '../types';
import { FileDropzone } from '../components/FileDropzone';
import { CameraCapture } from '../components/CameraCapture';
import { PredictionResult } from '../components/PredictionResult';
import { CorrectionForm } from '../components/CorrectionForm';
import { ErrorAlert } from '../components/ErrorAlert';
import { Spinner } from '../components/Spinner';

export const Home = () => {
  // État pour le backend
  const [status, setStatus] = useState<'loading' | 'online' | 'offline'>('loading');
  const [backendError, setBackendError] = useState<string | null>(null);

  // État pour la prédiction
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // État pour les informations du modèle (vraies données depuis le backend)
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [modelInfoLoading, setModelInfoLoading] = useState(true);
  const [modelInfoError, setModelInfoError] = useState<string | null>(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await health();
        setStatus('online');
      } catch (err) {
        setStatus('offline');
        setBackendError(err instanceof Error ? err.message : 'Backend non disponible');
      }
    };

    const fetchModelInfo = async () => {
      try {
        setModelInfoLoading(true);
        const info = await getModelInfo();
        setModelInfo(info);
        setModelInfoError(null);
      } catch (err) {
        setModelInfoError(err instanceof Error ? err.message : 'Erreur lors du chargement des informations du modèle');
        console.error('Erreur lors du chargement des informations du modèle:', err);
      } finally {
        setModelInfoLoading(false);
      }
    };

    checkHealth();
    fetchModelInfo();
  }, []);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setError(null);
    setPrediction(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handlePredict = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    try {
      const result = await predictImage(selectedFile);
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erreur lors de la prédiction');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCorrectionSuccess = () => {
    setSelectedFile(null);
    setImagePreview(null);
    setPrediction(null);
  };

  const handleReset = () => {
    setSelectedFile(null);
    setImagePreview(null);
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="space-y-8">
      {/* En-tête principal */}
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
          Banana Prediction
        </h1>
        <p className="mt-4 text-lg text-gray-600 dark:text-gray-400">
          Analysez vos images de bananes et obtenez des prédictions de maturité
        </p>
      </div>

      {/* Grille avec état du backend et informations du modèle */}
      <div className="mx-auto grid max-w-6xl gap-6 lg:grid-cols-2">
        {/* État du backend */}
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
              {backendError && <ErrorAlert error={backendError} />}
            </div>
          )}
        </div>

        {/* Informations du modèle (vraies données depuis le backend) */}
        <div className="rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
          <h2 className="mb-4 text-lg font-semibold text-gray-900 dark:text-white">
            Informations du modèle
          </h2>

          {modelInfoLoading && (
            <div className="flex items-center gap-3">
              <Spinner size="sm" />
              <span className="text-gray-600 dark:text-gray-400">Chargement des informations...</span>
            </div>
          )}

          {modelInfoError && (
            <ErrorAlert error={modelInfoError} />
          )}

          {modelInfo && !modelInfoLoading && (
            <dl className="space-y-2 text-sm">
              <div className="flex justify-between">
                <dt className="text-gray-600 dark:text-gray-400">Nom :</dt>
                <dd className="font-medium text-gray-900 dark:text-white">{modelInfo.name}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-600 dark:text-gray-400">Version :</dt>
                <dd className="font-medium text-gray-900 dark:text-white">{modelInfo.version}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-600 dark:text-gray-400">Taille d'image :</dt>
                <dd className="font-medium text-gray-900 dark:text-white">{modelInfo.imageSize}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-600 dark:text-gray-400">Format :</dt>
                <dd className="font-medium text-gray-900 dark:text-white">{modelInfo.format}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-600 dark:text-gray-400">Type :</dt>
                <dd className="font-medium text-gray-900 dark:text-white">{modelInfo.modelType}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-600 dark:text-gray-400">Device :</dt>
                <dd className="font-medium text-gray-900 dark:text-white">{modelInfo.device}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-600 dark:text-gray-400">Classes :</dt>
                <dd className="font-medium text-gray-900 dark:text-white">{modelInfo.classes.join(', ')}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-600 dark:text-gray-400">Modèle chargé :</dt>
                <dd className="font-medium text-gray-900 dark:text-white">
                  {modelInfo.modelLoaded ? '✅ Oui' : '❌ Non'}
                </dd>
              </div>
              {modelInfo.configuration?.mockPredictions && (
                <div className="flex justify-between">
                  <dt className="text-gray-600 dark:text-gray-400">Mode :</dt>
                  <dd className="font-medium text-orange-600 dark:text-orange-400">🧪 Test (prédictions simulées)</dd>
                </div>
              )}
              {modelInfo.error && (
                <div className="mt-2">
                  <ErrorAlert error={modelInfo.error} />
                </div>
              )}
            </dl>
          )}
        </div>
      </div>

      {/* Section de prédiction */}
      <div className="mx-auto max-w-4xl">
        <div className="rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
          <h2 className="mb-6 text-2xl font-semibold text-gray-900 dark:text-white">
            Faire une prédiction
          </h2>

          {error && (
            <div className="mb-6">
              <ErrorAlert error={error} onDismiss={() => setError(null)} />
            </div>
          )}

          {!imagePreview ? (
            <div className="space-y-4">
              <FileDropzone onFileSelect={handleFileSelect} disabled={isLoading} />
              <div className="relative">
                <div className="absolute inset-0 flex items-center" aria-hidden="true">
                  <div className="w-full border-t border-gray-300 dark:border-gray-600" />
                </div>
                <div className="relative flex justify-center">
                  <span className="bg-white px-3 text-sm text-gray-500 dark:bg-gray-800 dark:text-gray-400">
                    ou
                  </span>
                </div>
              </div>
              <CameraCapture onCapture={handleFileSelect} disabled={isLoading} />
            </div>
          ) : (
            <div className="space-y-4">
              <div className="relative overflow-hidden rounded-lg">
                <img
                  src={imagePreview}
                  alt="Aperçu de l'image sélectionnée"
                  className="mx-auto max-h-96 object-contain"
                />
              </div>
              {!prediction && (
                <div className="flex gap-3">
                  <button
                    onClick={handlePredict}
                    disabled={isLoading}
                    className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-primary-600 px-4 py-2.5 font-medium text-white hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50"
                  >
                    {isLoading ? (
                      <>
                        <Spinner size="sm" />
                        Analyse en cours...
                      </>
                    ) : (
                      'Prédire'
                    )}
                  </button>
                  <button
                    onClick={handleReset}
                    disabled={isLoading}
                    className="rounded-lg border border-gray-300 px-4 py-2.5 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 dark:border-gray-600 dark:hover:bg-gray-700"
                  >
                    Nouvelle image
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Résultats de prédiction et correction */}
      {prediction && (
        <div className="mx-auto max-w-4xl space-y-6">
          <PredictionResult prediction={prediction} />
          <CorrectionForm
            prediction={prediction}
            onSuccess={handleCorrectionSuccess}
            onReset={handleReset}
          />
        </div>
      )}
    </div>
  );
};
