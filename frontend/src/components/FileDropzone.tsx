import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { isAvifFile, processImageToJpg } from '../utils/imageUtils';

interface FileDropzoneProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export const FileDropzone = ({ onFileSelect, disabled }: FileDropzoneProps) => {
  const [isProcessing, setIsProcessing] = useState(false);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];

        if (isAvifFile(file)) {
          alert('Les fichiers AVIF ne sont pas acceptés. Veuillez utiliser JPG, PNG ou WebP.');
          return;
        }

        try {
          setIsProcessing(true);
          const processedFile = await processImageToJpg(file);
          onFileSelect(processedFile);
        } catch (error) {
          console.error('Erreur lors du traitement de l\'image:', error);
          alert('Erreur lors du traitement de l\'image');
        } finally {
          setIsProcessing(false);
        }
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp'],
    },
    multiple: false,
    disabled: disabled || isProcessing,
  });

  return (
    <div
      {...getRootProps()}
      className={`
        flex min-h-[240px] cursor-pointer flex-col items-center justify-center
        rounded-lg border-2 border-dashed p-8 text-center transition-colors
        ${
          isDragActive
            ? 'border-primary-500 bg-primary-50 dark:bg-primary-950'
            : 'border-gray-300 hover:border-gray-400 dark:border-gray-600 dark:hover:border-gray-500'
        }
        ${disabled || isProcessing ? 'cursor-not-allowed opacity-50' : ''}
        focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2
      `}
      role="button"
      aria-label="Zone de dépôt de fichier"
      tabIndex={0}
    >
      <input {...getInputProps()} aria-label="Sélectionner un fichier image" />

      {isProcessing ? (
        <>
          <div className="mb-4 h-12 w-12 animate-spin rounded-full border-4 border-primary-200 border-t-primary-600"></div>
          <p className="text-sm text-gray-600 dark:text-gray-300">
            Traitement de l'image...
          </p>
        </>
      ) : (
        <>
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-primary-100 dark:bg-primary-900">
            <svg
              className="h-6 w-6 text-primary-600 dark:text-primary-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
          </div>

          <div className="space-y-2">
            <p className="text-lg font-medium text-gray-900 dark:text-gray-100">
              {isDragActive ? "Déposez l'image ici" : "Glissez-déposez votre image"}
            </p>

            <p className="text-sm text-gray-500 dark:text-gray-400">
              ou <span className="font-medium text-primary-600 dark:text-primary-400">cliquez pour parcourir</span>
            </p>

            <p className="text-xs text-gray-400 dark:text-gray-500">
              JPG, PNG, WebP uniquement (pas d'AVIF)
            </p>
          </div>
        </>
      )}
    </div>
  );
};
