/**
 * Utilitaires pour la manipulation d'images côté frontend
 */

export interface ImageResizeOptions {
  maxWidth?: number;
  maxHeight?: number;
  quality?: number; // 0.1 à 1.0 pour JPEG
  format?: 'jpeg' | 'png' | 'webp';
}

const DEFAULT_OPTIONS: Required<ImageResizeOptions> = {
  maxWidth: 4096,
  maxHeight: 4096,
  quality: 0.9,
  format: 'jpeg'
};

/**
 * Redimensionne une image si elle dépasse les dimensions maximales
 */
export const resizeImageFile = async (
  file: File,
  options: ImageResizeOptions = {}
): Promise<File> => {
  const opts = { ...DEFAULT_OPTIONS, ...options };

  return new Promise((resolve, reject) => {
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      reject(new Error('Impossible de créer le contexte canvas'));
      return;
    }

    img.onload = () => {
      // Calculer les nouvelles dimensions en conservant le ratio
      const { width: newWidth, height: newHeight } = calculateNewDimensions(
        img.width,
        img.height,
        opts.maxWidth,
        opts.maxHeight
      );

      // Si l'image est déjà dans les limites, renvoyer le fichier original
      if (newWidth === img.width && newHeight === img.height) {
        resolve(file);
        return;
      }

      // Redimensionner l'image
      canvas.width = newWidth;
      canvas.height = newHeight;

      // Dessiner l'image redimensionnée
      ctx.drawImage(img, 0, 0, newWidth, newHeight);

      // Convertir en blob puis en File
      canvas.toBlob(
        (blob) => {
          if (!blob) {
            reject(new Error('Erreur lors de la conversion de l\'image'));
            return;
          }

          const resizedFile = new File(
            [blob],
            file.name,
            {
              type: `image/${opts.format}`,
              lastModified: Date.now()
            }
          );

          console.log(`[ImageUtils] Image redimensionnée de ${img.width}x${img.height} à ${newWidth}x${newHeight}`);
          resolve(resizedFile);
        },
        `image/${opts.format}`,
        opts.quality
      );
    };

    img.onerror = () => {
      reject(new Error('Erreur lors du chargement de l\'image'));
    };

    // Charger l'image
    img.src = URL.createObjectURL(file);
  });
};

/**
 * Calcule les nouvelles dimensions en conservant le ratio d'aspect
 */
const calculateNewDimensions = (
  originalWidth: number,
  originalHeight: number,
  maxWidth: number,
  maxHeight: number
): { width: number; height: number } => {
  // Si l'image est déjà dans les limites
  if (originalWidth <= maxWidth && originalHeight <= maxHeight) {
    return { width: originalWidth, height: originalHeight };
  }

  // Calculer le ratio de redimensionnement
  const widthRatio = maxWidth / originalWidth;
  const heightRatio = maxHeight / originalHeight;
  const ratio = Math.min(widthRatio, heightRatio);

  return {
    width: Math.floor(originalWidth * ratio),
    height: Math.floor(originalHeight * ratio)
  };
};

/**
 * Vérifie si une image dépasse les dimensions maximales
 */
export const checkImageDimensions = async (file: File): Promise<{ width: number; height: number; needsResize: boolean }> => {
  return new Promise((resolve, reject) => {
    const img = new Image();

    img.onload = () => {
      const needsResize = img.width > DEFAULT_OPTIONS.maxWidth || img.height > DEFAULT_OPTIONS.maxHeight;
      resolve({
        width: img.width,
        height: img.height,
        needsResize
      });
      URL.revokeObjectURL(img.src);
    };

    img.onerror = () => {
      reject(new Error('Impossible de lire les dimensions de l\'image'));
    };

    img.src = URL.createObjectURL(file);
  });
};
