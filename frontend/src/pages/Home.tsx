import { useState, useEffect, useRef } from 'react';
import { health, predictImage, getModelInfo, type ModelInfo } from '../api/endpoints';
import { PredictResponse } from '../types';
import { FileDropzone } from '../components/FileDropzone';
import { CameraCapture } from '../components/CameraCapture';
import { PredictionResult } from '../components/PredictionResult';
import { CorrectionForm } from '../components/CorrectionForm';
import { ErrorAlert } from '../components/ErrorAlert';
import { Spinner } from '../components/Spinner';
import { useSettings } from '../store/useSettings';
import { useTranslation } from '../utils/i18n';

export const Home = () => {
  const { language } = useSettings();
  const t = useTranslation(language);

  // État pour le backend
  const [status, setStatus] = useState<'loading' | 'online' | 'offline'>('loading');

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
  const [showModelInfo, setShowModelInfo] = useState(false);
  const modelInfoRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await health();
        setStatus('online');
      } catch (err) {
        setStatus('offline');
        setError(err instanceof Error ? err.message : t.errorBackendUnavailable);
      }
    };

    const fetchModelInfo = async () => {
      try {
        setModelInfoLoading(true);
        const info = await getModelInfo();
        setModelInfo(info);
        setModelInfoError(null);
      } catch (err) {
        setModelInfoError(err instanceof Error ? err.message : t.errorModelInfoLoading);
        console.error('Erreur lors du chargement des informations du modèle:', err);
      } finally {
        setModelInfoLoading(false);
      }
    };

    checkHealth();
    fetchModelInfo();
  }, [t]);

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
      setError(err instanceof Error ? err.message : t.errorPrediction);
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
      {/* En-tête principal avec design moderne */}
      <div className="text-center">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-yellow-600 via-orange-600 to-red-500 bg-clip-text text-transparent dark:from-yellow-400 dark:via-orange-400 dark:to-red-400 mb-4">
          {t.mainTitle}
        </h1>
        <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
          {t.welcomeSubtitle}
        </p>
      </div>

      {/* Section prédiction + menu déroulant infos modèle/status */}
      <div className="mx-auto max-w-4xl">
        <div className="rounded-2xl border border-yellow-200/50 bg-white/60 backdrop-blur-sm p-8 shadow-lg dark:border-gray-700/50 dark:bg-gray-800/60">
          {/* Header avec titre à gauche et bouton infos à droite */}
          <div className="flex items-center justify-between mb-6 relative">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-green-100 rounded-lg dark:bg-green-900/30">
                <svg className="w-6 h-6 text-green-600 dark:text-green-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
                </svg>
              </div>
              <h2 className="text-2xl font-semibold text-gray-900 dark:text-white">
                {t.predictionTitle}
              </h2>
            </div>

            {/* Bouton menu déroulant infos modèle/status à droite */}
            <div className="relative">
              <button
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-sm font-medium shadow"
                onClick={() => setShowModelInfo((v) => !v)}
                aria-expanded={showModelInfo}
                aria-controls="model-info-panel"
              >
                {/* pastille status */}
                {status === 'online' && (
                  <span className="w-2.5 h-2.5 rounded-full bg-green-500 animate-pulse" />
                )}
                {status === 'offline' && (
                  <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
                )}
                {status === 'loading' && (
                  <Spinner size="sm" />
                )}
                {t.modelInfoTitle}
                <svg className={`w-4 h-4 ml-1 transition-transform ${showModelInfo ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {showModelInfo && (
                <div
                  id="model-info-panel"
                  ref={modelInfoRef}
                  className="absolute right-0 mt-2 w-80 max-w-[90vw] z-10 p-4 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900/90 shadow-2xl backdrop-blur"
                >
                  {/* Status en toutes lettres */}
                  <div className="flex items-center gap-2 mb-3 text-xs">
                    {status === 'loading' && (
                      <>
                        <Spinner size="sm" />
                        <span className="text-gray-600 dark:text-gray-400">{t.statusChecking}</span>
                      </>
                    )}
                    {status === 'online' && (
                      <span className="inline-flex items-center gap-1 text-green-700 dark:text-green-400">
                        <span className="w-2.5 h-2.5 bg-green-500 rounded-full animate-pulse inline-block" />
                        {t.statusOnline}
                      </span>
                    )}
                    {status === 'offline' && (
                      <span className="inline-flex items-center gap-1 text-red-700 dark:text-red-400">
                        <span className="w-2.5 h-2.5 bg-red-500 rounded-full inline-block" />
                        {t.statusOffline}
                      </span>
                    )}
                  </div>

                  {/* Infos modèle dynamiques */}
                  {modelInfoLoading && (
                    <div className="flex items-center gap-3">
                      <Spinner size="sm" />
                      <span className="text-gray-600 dark:text-gray-400">{t.modelInfoLoading}</span>
                    </div>
                  )}
                  {modelInfoError && <ErrorAlert error={modelInfoError} />}
                  {modelInfo && !modelInfoLoading && (
                    <dl className="space-y-2 text-sm">
                      <div className="flex justify-between items-center">
                        <dt className="text-gray-600 dark:text-gray-400">{t.modelName} :</dt>
                        <dd className="font-medium text-gray-900 dark:text-white">{modelInfo.name}</dd>
                      </div>
                      <div className="flex justify-between items-center">
                        <dt className="text-gray-600 dark:text-gray-400">{t.modelVersion} :</dt>
                        <dd className="font-medium text-gray-900 dark:text-white">{modelInfo.version}</dd>
                      </div>
                      <div className="flex justify-between items-center">
                        <dt className="text-gray-600 dark:text-gray-400">{t.modelImageSize} :</dt>
                        <dd className="font-medium text-gray-900 dark:text-white">{modelInfo.imageSize}</dd>
                      </div>
                      <div className="flex justify-between items-center">
                        <dt className="text-gray-600 dark:text-gray-400">{t.modelFormat} :</dt>
                        <dd className="font-medium text-gray-900 dark:text-white">
                          <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                            {modelInfo.format}
                          </span>
                        </dd>
                      </div>
                      <div className="flex justify-between items-center">
                        <dt className="text-gray-600 dark:text-gray-400">{t.modelType} :</dt>
                        <dd className="font-medium text-gray-900 dark:text-white">{modelInfo.modelType}</dd>
                      </div>
                      <div className="flex justify-between items-center">
                        <dt className="text-gray-600 dark:text-gray-400">{t.modelDevice} :</dt>
                        <dd className="font-medium text-gray-900 dark:text-white">{modelInfo.device}</dd>
                      </div>
                      <div className="flex justify-between items-start">
                        <dt className="text-gray-600 dark:text-gray-400">{t.modelClasses} :</dt>
                        <dd className="font-medium text-gray-900 dark:text-white text-right">
                          <div className="flex flex-wrap gap-1 justify-end">
                            {modelInfo.classes.map((cls) => (
                              <span key={cls} className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
                                {cls}
                              </span>
                            ))}
                          </div>
                        </dd>
                      </div>
                      <div className="flex justify-between items-center">
                        <dt className="text-gray-600 dark:text-gray-400">{t.modelLoaded} :</dt>
                        <dd className="font-medium text-gray-900 dark:text-white">
                          {modelInfo.modelLoaded ? (
                            <span className="inline-flex items-center gap-2 text-green-600 dark:text-green-400">
                              <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                              {t.yes}
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-2 text-red-600 dark:text-red-400">
                              <span className="w-2 h-2 bg-red-500 rounded-full"></span>
                              {t.no}
                            </span>
                          )}
                        </dd>
                      </div>
                      {modelInfo.configuration?.mockPredictions && (
                        <div className="flex justify-between items-center">
                          <dt className="text-gray-600 dark:text-gray-400">{t.modelMode} :</dt>
                          <dd className="font-medium text-orange-600 dark:text-orange-400">{t.modelModeTest}</dd>
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
              )}
            </div>
          </div>

          {/* Section prédiction image/caméra */}
          <div className="space-y-6">
            {error && (
              <div className="mb-6">
                <ErrorAlert error={error} onDismiss={() => setError(null)} />
              </div>
            )}

            {!imagePreview ? (
              <div className="space-y-6">
                <FileDropzone onFileSelect={handleFileSelect} disabled={isLoading} />
                <div className="relative">
                  <div className="absolute inset-0 flex items-center" aria-hidden="true">
                    <div className="w-full border-t border-gray-300 dark:border-gray-600" />
                  </div>
                  <div className="relative flex justify-center">
                    <span className="bg-white/80 backdrop-blur-sm px-4 py-2 text-sm text-gray-500 dark:bg-gray-800/80 dark:text-gray-400 rounded-full">
                      {t.predictionOr}
                    </span>
                  </div>
                </div>
                <CameraCapture onCapture={handleFileSelect} disabled={isLoading} />
              </div>
            ) : (
              <div className="space-y-6">
                <div className="relative overflow-hidden rounded-2xl bg-gray-100 dark:bg-gray-700">
                  <img
                    src={imagePreview}
                    alt="Aperçu de l'image sélectionnée"
                    className="mx-auto max-h-96 object-contain"
                  />
                </div>

                {/* Bouton "Nouvelle image" sous l'image */}
                {prediction && (
                  <div className="flex justify-center">
                    <button
                      onClick={handleReset}
                      className="w-full max-w-md rounded-xl border-2 border-gray-300 px-6 py-3 font-semibold text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-700 transition-all"
                    >
                      Nouvelle image
                    </button>
                  </div>
                )}

                {!prediction && (
                  <div className="flex gap-4">
                    <button
                      onClick={handlePredict}
                      disabled={isLoading}
                      className="flex flex-1 items-center justify-center gap-3 rounded-xl bg-gradient-to-r from-green-500 to-emerald-600 px-6 py-3 font-semibold text-white shadow-lg hover:from-green-600 hover:to-emerald-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 disabled:opacity-50 transition-all"
                    >
                      {isLoading ? (
                        <>
                          <Spinner size="sm" />
                          {t.predictionAnalyzing}
                        </>
                      ) : (
                        <>
                          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                          </svg>
                          {t.predictionPredict}
                        </>
                      )}
                    </button>
                    <button
                      onClick={handleReset}
                      disabled={isLoading}
                      className="rounded-xl border-2 border-gray-300 px-6 py-3 font-semibold text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-700 transition-all"
                    >
                      {t.predictionNewImage}
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Résultats de prédiction et correction */}
      {prediction && (
        <div className="mx-auto max-w-4xl space-y-6">

          <PredictionResult prediction={prediction} />
          <CorrectionForm
            prediction={prediction}
          />
        </div>
      )}
    </div>
  );
};
