import { useState } from 'react';
import { FileDropzone } from '../components/FileDropzone';
import { CameraCapture } from '../components/CameraCapture';
import { PredictionResult } from '../components/PredictionResult';
import { CorrectionForm } from '../components/CorrectionForm';
import { ErrorAlert } from '../components/ErrorAlert';
import { Spinner } from '../components/Spinner';
import { predictImage } from '../api/endpoints';
import { PredictResponse } from '../types';

type Tab = 'file' | 'camera';

export const Predict = () => {
  const [activeTab, setActiveTab] = useState<Tab>('file');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Prédiction</h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Uploadez une image ou prenez une photo pour obtenir une prédiction
        </p>
      </div>

      <div className="rounded-lg border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800">
        <div className="flex border-b border-gray-200 dark:border-gray-700">
          <button
            onClick={() => setActiveTab('file')}
            className={`flex-1 px-6 py-3 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary-500 ${
              activeTab === 'file'
                ? 'border-b-2 border-primary-600 text-primary-600 dark:border-primary-400 dark:text-primary-400'
                : 'text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-200'
            }`}
            role="tab"
            aria-selected={activeTab === 'file'}
            aria-controls="file-panel"
          >
            📁 Fichier
          </button>
          <button
            onClick={() => setActiveTab('camera')}
            className={`flex-1 px-6 py-3 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary-500 ${
              activeTab === 'camera'
                ? 'border-b-2 border-primary-600 text-primary-600 dark:border-primary-400 dark:text-primary-400'
                : 'text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-200'
            }`}
            role="tab"
            aria-selected={activeTab === 'camera'}
            aria-controls="camera-panel"
          >
            📷 Caméra
          </button>
        </div>

        <div className="p-6">
          {error && (
            <div className="mb-6">
              <ErrorAlert error={error} onDismiss={() => setError(null)} />
            </div>
          )}

          <div
            id="file-panel"
            role="tabpanel"
            aria-labelledby="file-tab"
            hidden={activeTab !== 'file'}
          >
            {!imagePreview ? (
              <FileDropzone onFileSelect={handleFileSelect} disabled={isLoading} />
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
                      Annuler
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>

          <div
            id="camera-panel"
            role="tabpanel"
            aria-labelledby="camera-tab"
            hidden={activeTab !== 'camera'}
          >
            <CameraCapture onCapture={handleFileSelect} disabled={isLoading} />
          </div>
        </div>
      </div>

      {prediction && (
        <div className="space-y-6">
          <PredictionResult prediction={prediction} />
          <CorrectionForm prediction={prediction} onSuccess={handleCorrectionSuccess} />
        </div>
      )}
    </div>
  );
};
