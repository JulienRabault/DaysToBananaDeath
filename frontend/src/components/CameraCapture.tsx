import { useState, useRef, useEffect } from 'react';
import { ErrorAlert } from './ErrorAlert';
import { useSettings } from '../store/useSettings';
import { useTranslation } from '../utils/i18n';

interface CameraCaptureProps {
  onCapture: (file: File) => void;
  disabled?: boolean;
}

export const CameraCapture = ({ onCapture, disabled }: CameraCaptureProps) => {
  const { language } = useSettings();
  const t = useTranslation(language);

  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const checkCameraPermission = async () => {
    try {
      const result = await navigator.permissions.query({ name: 'camera' as PermissionName });
      setHasPermission(result.state === 'granted');

      result.addEventListener('change', () => {
        setHasPermission(result.state === 'granted');
      });
    } catch (err) {
      console.log('Permission API not supported:', err);
      setHasPermission(null);
    }
  };

  const startCamera = async () => {
    try {
      setError(null);

      // Vérifier d'abord si les médias sont supportés
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(t.cameraNotSupported);
      }

      const constraints = {
        video: {
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false,
      };

      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(mediaStream);
      setHasPermission(true);

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (err: any) {
      console.error('Camera error:', err);
      let errorMessage = t.cameraPermissionError;

      if (err.name === 'NotAllowedError') {
        errorMessage = t.cameraPermissionDenied;
      } else if (err.name === 'NotFoundError') {
        errorMessage = t.cameraNotFound;
      } else if (err.name === 'NotSupportedError') {
        errorMessage = t.cameraNotSupported;
      } else if (err.name === 'NotReadableError') {
        errorMessage = t.cameraInUse;
      }

      setError(errorMessage);
      setHasPermission(false);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
  };

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `capture-${Date.now()}.jpg`, { type: 'image/jpeg' });
        onCapture(file);
        setIsCapturing(true);
        stopCamera();
      }
    }, 'image/jpeg');
  };

  const resetCapture = () => {
    setIsCapturing(false);
    startCamera();
  };

  useEffect(() => {
    checkCameraPermission();
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="space-y-4">
      {error && <ErrorAlert error={error} onDismiss={() => setError(null)} />}

      <div className="relative overflow-hidden rounded-lg bg-gray-900">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full ${stream && !isCapturing ? 'block' : 'hidden'}`}
          aria-label={t.cameraVideoLabel}
        />
        <canvas ref={canvasRef} className="hidden" />

        {!stream && !isCapturing && (
          <div className="flex min-h-[320px] items-center justify-center bg-gray-100 dark:bg-gray-800">
            <button
              onClick={startCamera}
              disabled={disabled}
              className="rounded-lg bg-primary-600 px-6 py-3 text-white hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50"
              aria-label={t.cameraStartButton}
            >
              <svg
                className="mx-auto mb-2 h-8 w-8"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"
                />
              </svg>
              {t.cameraStartButton}
            </button>
          </div>
        )}
      </div>

      {stream && !isCapturing && (
        <div className="flex gap-3">
          <button
            onClick={capturePhoto}
            disabled={disabled}
            className="flex-1 rounded-lg bg-primary-600 px-4 py-2 text-white hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50"
          >
            {t.cameraCaptureButton}
          </button>
          <button
            onClick={stopCamera}
            className="rounded-lg border border-gray-300 px-4 py-2 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 dark:border-gray-600 dark:hover:bg-gray-800"
          >
            {t.predictionCancel}
          </button>
        </div>
      )}

      {isCapturing && (
        <button
          onClick={resetCapture}
          className="w-full rounded-lg border border-gray-300 px-4 py-2 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 dark:border-gray-600 dark:hover:bg-gray-800"
        >
          {t.cameraNewPhotoButton}
        </button>
      )}
    </div>
  );
};
