import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

interface FileDropzoneProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export const FileDropzone = ({ onFileSelect, disabled }: FileDropzoneProps) => {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onFileSelect(acceptedFiles[0]);
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp'], // Remis .webp - conversion backend
    },
    multiple: false,
    disabled,
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
        ${disabled ? 'cursor-not-allowed opacity-50' : ''}
        focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2
      `}
      role="button"
      aria-label="Zone de dépôt de fichier"
      tabIndex={0}
    >
      <input {...getInputProps()} aria-label="Sélectionner un fichier image" />
      <svg
        className="mb-4 h-12 w-12 text-gray-400"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        aria-hidden="true"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
        />
      </svg>
      {isDragActive ? (
        <p className="text-sm text-primary-600 dark:text-primary-400">
          Déposez l&apos;image ici...
        </p>
      ) : (
        <>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Glissez-déposez une image ici, ou cliquez pour sélectionner
          </p>
          <p className="mt-2 text-xs text-gray-500 dark:text-gray-500">
            PNG, JPG, JPEG, WEBP
          </p>
        </>
      )}
    </div>
  );
};
