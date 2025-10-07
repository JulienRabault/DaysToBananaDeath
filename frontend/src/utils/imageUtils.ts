export function isAvifFile(file: File): boolean {
  return file.type === 'image/avif' || file.name.toLowerCase().endsWith('.avif');
}

export function processImageToJpg(file: File): Promise<File> {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
      canvas.width = 512;
      canvas.height = 512;

      ctx!.drawImage(img, 0, 0, 512, 512);

      canvas.toBlob((blob) => {
        if (blob) {
          const newFile = new File([blob], file.name.replace(/\.[^/.]+$/, '.jpg'), {
            type: 'image/jpeg'
          });

          resolve(newFile);
        } else {
          reject(new Error('Erreur lors de la conversion'));
        }
      }, 'image/jpeg', 0.9);
    };

    img.onerror = () => reject(new Error('Erreur lors du chargement de l\'image'));
    img.src = URL.createObjectURL(file);
  });
}
