/**
 * Utilitaires pour la gestion des erreurs API
 */

export interface ApiError {
  message: string;
  details?: string;
  status?: number;
  code?: string;
}

/**
 * Parse une erreur de l'API et retourne un message utilisateur approprié
 */
export const parseApiError = (error: any): ApiError => {
  // Erreur réseau/connexion
  if (error.name === 'NetworkError' || error.message === 'Erreur de connexion') {
    return {
      message: 'Problème de connexion',
      details: 'Impossible de joindre le serveur. Vérifiez votre connexion internet.',
      code: 'NETWORK_ERROR'
    };
  }

  // Erreur HTTP avec réponse JSON
  if (error.status && error.details) {
    const status = error.status;
    let message = 'Erreur du serveur';
    let details = error.details;

    // Essayer de parser le détail si c'est un objet
    if (typeof details === 'object' && details.detail) {
      details = details.detail;
    }

    switch (status) {
      case 400:
        message = 'Données invalides';
        // Messages spécifiques pour certaines erreurs 400
        if (typeof details === 'string') {
          if (details.includes('Image too large')) {
            message = 'Image trop grande';
            details = 'L\'image dépasse la taille maximale autorisée. Elle sera automatiquement redimensionnée.';
          } else if (details.includes('File too large')) {
            message = 'Fichier trop volumineux';
            details = 'La taille du fichier dépasse la limite autorisée (10 MB maximum).';
          } else if (details.includes('File extension not allowed')) {
            message = 'Format de fichier non supporté';
            details = 'Formats autorisés : JPG, PNG, WEBP, AVIF, GIF, BMP, TIFF';
          } else if (details.includes('not a valid image')) {
            message = 'Fichier image invalide';
            details = 'Le fichier sélectionné n\'est pas une image valide ou est corrompu.';
          } else if (details.includes('rate limit')) {
            message = 'Trop de requêtes';
            details = 'Vous avez effectué trop de prédictions. Veuillez patienter avant de réessayer.';
          }
        }
        break;

      case 401:
        message = 'Non autorisé';
        details = 'Vous n\'êtes pas autorisé à effectuer cette action.';
        break;

      case 403:
        message = 'Accès interdit';
        details = 'L\'accès à cette ressource est interdit.';
        break;

      case 404:
        message = 'Service non trouvé';
        details = 'Le service demandé n\'existe pas ou n\'est pas disponible.';
        break;

      case 413:
        message = 'Fichier trop volumineux';
        details = 'La taille du fichier dépasse la limite autorisée par le serveur.';
        break;

      case 422:
        message = 'Données incorrectes';
        details = 'Les données envoyées sont incorrectes ou incomplètes.';
        break;

      case 429:
        message = 'Trop de requêtes';
        details = 'Vous avez effectué trop de requêtes. Veuillez patienter avant de réessayer.';
        break;

      case 500:
        message = 'Erreur du serveur';
        details = 'Une erreur interne s\'est produite sur le serveur. Veuillez réessayer plus tard.';
        break;

      case 502:
        message = 'Service indisponible';
        details = 'Le serveur est temporairement indisponible. Veuillez réessayer plus tard.';
        break;

      case 503:
        message = 'Service en maintenance';
        details = 'Le service est temporairement en maintenance. Veuillez réessayer plus tard.';
        break;

      default:
        message = `Erreur HTTP ${status}`;
    }

    return {
      message,
      details,
      status,
      code: `HTTP_${status}`
    };
  }

  // Erreur timeout
  if (error.name === 'TimeoutError' || error.message?.includes('timeout')) {
    return {
      message: 'Délai d\'attente dépassé',
      details: 'La requête a pris trop de temps. Le serveur est peut-être surchargé.',
      code: 'TIMEOUT_ERROR'
    };
  }

  // Erreur CORS
  if (error.message?.includes('CORS') || error.message?.includes('blocked')) {
    return {
      message: 'Problème de configuration',
      details: 'Erreur de configuration du serveur. Contactez l\'administrateur.',
      code: 'CORS_ERROR'
    };
  }

  // Erreur de parsing JSON
  if (error.name === 'SyntaxError' || error.message?.includes('JSON')) {
    return {
      message: 'Réponse invalide du serveur',
      details: 'Le serveur a renvoyé une réponse dans un format incorrect.',
      code: 'PARSE_ERROR'
    };
  }

  // Erreur générique
  return {
    message: 'Erreur inattendue',
    details: error.message || 'Une erreur inattendue s\'est produite.',
    code: 'UNKNOWN_ERROR'
  };
};

/**
 * Formatte un message d'erreur pour l'affichage utilisateur
 */
export const formatErrorMessage = (error: ApiError): string => {
  if (error.details && error.details !== error.message) {
    return `${error.message}: ${error.details}`;
  }
  return error.message;
};

/**
 * Vérifie si une erreur est temporaire et peut être réessayée
 */
export const isRetryableError = (error: ApiError): boolean => {
  const retryableCodes = [
    'NETWORK_ERROR',
    'TIMEOUT_ERROR',
    'HTTP_500',
    'HTTP_502',
    'HTTP_503'
  ];

  return retryableCodes.includes(error.code || '');
};
