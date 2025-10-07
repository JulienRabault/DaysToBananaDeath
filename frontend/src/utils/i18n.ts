export type Language = 'fr' | 'en';

export interface Translations {
  // Navigation et layout
  appTitle: string;
  appSubtitle: string;

  // Page principale
  welcomeTitle: string;
  welcomeSubtitle: string;
  mainTitle: string; // Nouveau pour le titre principal

  // États du backend
  backendStatusTitle: string;
  statusChecking: string;
  statusOnline: string;
  statusOffline: string;

  // Informations du modèle
  modelInfoTitle: string;
  modelInfoLoading: string;
  modelName: string;
  modelVersion: string;
  modelImageSize: string;
  modelFormat: string;
  modelType: string;
  modelDevice: string;
  modelClasses: string;
  modelLoaded: string;
  modelMode: string;
  modelModeTest: string;

  // Prédiction
  predictionTitle: string;
  predictionSubtitle: string;
  predictionAnalyzing: string;
  predictionPredict: string;
  predictionNewImage: string;
  predictionCancel: string;
  predictionOr: string;
  predictionImageAlt: string;
  predictionError: string;

  // Caméra - nouvelles traductions
  cameraStartButton: string;
  cameraCaptureButton: string;
  cameraNewPhotoButton: string;
  cameraVideoLabel: string;
  cameraPermissionError: string;
  cameraPermissionDenied: string;
  cameraNotFound: string;
  cameraNotSupported: string;
  cameraInUse: string;

  correctionTitle: string;
  correctionSubmit: string;
  correctionReset: string;
  correctionBananaQuestion: string;
  correctionYesBanana: string;
  correctionNoBanana: string;
  correctionDurationTitle: string;
  correctionDurationSubtitle: string;
  correctionDays: string;
  correctionDay: string;
  correctionStateImmediate: string;
  correctionStateOptimal: string;
  correctionStateConservation: string;
  correctionStatePredicted: string;
  correctionSendButton: string;
  correctionSending: string;
  correctionSuccessTitle: string;
  correctionSuccessMessage: string;
  correctionNewPredictionMessage: string;
  correctionCurrentPrediction: string;

  stateVeryRipe: string;
  stateRipe: string;
  stateGoodRipeness: string;
  stateMediumRipeness: string;
  stateStillGreen: string;

  resultTitle: string;
  resultConfidence: string;
  resultTopPredictions: string;
  resultImageKey: string;

  // États généraux
  yes: string;
  no: string;
  loading: string;
  error: string;
  success: string;

  // Messages d'erreur
  errorBackendUnavailable: string;
  errorModelInfoLoading: string;
  errorPrediction: string;

  // Classes de bananes
  classUnripe: string;
  classRipe: string;
  classOverripe: string;
  classRotten: string;
  classUnknowns: string;

  // Thème
  themeLight: string;
  themeDark: string;
  themeSystem: string;
  themeToggle: string;

  // Langue
  languageSwitch: string;
  french: string;
  english: string;
}

export const translations: Record<Language, Translations> = {
  fr: {
    // Navigation et layout
    appTitle: "Durée de vie des bananes",
    appSubtitle: "Prédiction et correction d'images",

    // Page principale
    welcomeTitle: "Durée de vie des bananes",
    welcomeSubtitle: "Analysez vos images de bananes et obtenez des prédictions de maturité",
    mainTitle: "Prédiction de maturité des bananes", // Titre principal

    // États du backend
    backendStatusTitle: "État du backend",
    statusChecking: "Vérification en cours...",
    statusOnline: "En ligne",
    statusOffline: "Hors ligne",

    // Informations du modèle
    modelInfoTitle: "Informations du modèle",
    modelInfoLoading: "Chargement des informations...",
    modelName: "Nom",
    modelVersion: "Version",
    modelImageSize: "Taille d'image",
    modelFormat: "Format",
    modelType: "Type",
    modelDevice: "Device",
    modelClasses: "Classes",
    modelLoaded: "Modèle chargé",
    modelMode: "Mode",
    modelModeTest: "🧪 Test (prédictions simulées)",

    // Prédiction
    predictionTitle: "Faire une prédiction",
    predictionSubtitle: "Uploadez une image pour obtenir une prédiction",
    predictionAnalyzing: "Analyse en cours...",
    predictionPredict: "Prédire",
    predictionNewImage: "Nouvelle image",
    predictionCancel: "Annuler",
    predictionOr: "ou",
    predictionImageAlt: "Image de la banane pour la prédiction",
    predictionError: "Erreur lors de la prédiction",

    // Caméra - nouvelles traductions
    cameraStartButton: "Démarrer la caméra",
    cameraCaptureButton: "Capturer",
    cameraNewPhotoButton: "Nouvelle photo",
    cameraVideoLabel: "Enregistrement vidéo",
    cameraPermissionError: "Erreur de permission de la caméra",
    cameraPermissionDenied: "Permission de caméra refusée",
    cameraNotFound: "Caméra non trouvée",
    cameraNotSupported: "Caméra non supportée",
    cameraInUse: "Caméra déjà utilisée",

    // Correction - nouvelles traductions
    correctionTitle: "Soumettre une correction",
    correctionSubmit: "Soumettre",
    correctionReset: "Recommencer",
    correctionBananaQuestion: "La prédiction de la banane est-elle correcte ?",
    correctionYesBanana: "Oui, c'est une banane",
    correctionNoBanana: "Non, ce n'est pas une banane",
    correctionDurationTitle: "Durée de conservation",
    correctionDurationSubtitle: "Combien de temps la banane a-t-elle été conservée ?",
    correctionDays: "jours",
    correctionDay: "jour",
    correctionStateImmediate: "État immédiat",
    correctionStateOptimal: "État optimal",
    correctionStateConservation: "État de conservation",
    correctionStatePredicted: "État prédit",
    correctionSendButton: "Envoyer la correction",
    correctionSending: "Envoi de la correction...",
    correctionSuccessTitle: "Correction soumise",
    correctionSuccessMessage: "Merci pour votre contribution !",
    correctionNewPredictionMessage: "Une nouvelle prédiction sera effectuée.",
    correctionCurrentPrediction: "Prédiction actuelle : {}",

    // États de maturité détaillés
    stateVeryRipe: "Très mûre",
    stateRipe: "Mûre",
    stateGoodRipeness: "Bonne maturité",
    stateMediumRipeness: "Maturité moyenne",
    stateStillGreen: "Encore verte",

    // Résultats
    resultTitle: "Résultat de la prédiction",
    resultConfidence: "de confiance",
    resultTopPredictions: "Top prédictions",
    resultImageKey: "Clé d'image",

    // États généraux
    yes: "Oui",
    no: "Non",
    loading: "Chargement",
    error: "Erreur",
    success: "Succès",

    // Messages d'erreur
    errorBackendUnavailable: "Backend non disponible",
    errorModelInfoLoading: "Erreur lors du chargement des informations du modèle",
    errorPrediction: "Erreur lors de la prédiction",

    // Classes de bananes
    classUnripe: "Pas mûre",
    classRipe: "Mûre",
    classOverripe: "Trop mûre",
    classRotten: "Pourrie",
    classUnknowns: "Inconnue",

    // Thème
    themeLight: "Clair",
    themeDark: "Sombre",
    themeSystem: "Système",
    themeToggle: "Changer le thème",

    // Langue
    languageSwitch: "Changer la langue",
    french: "Français",
    english: "English",
  },

  en: {
    // Navigation et layout
    appTitle: "Days to banana death",
    appSubtitle: "Image prediction and correction",

    // Page principale
    welcomeTitle: "Days to banana death",
    welcomeSubtitle: "Analyze your banana images and get ripeness predictions",
    mainTitle: "Banana ripeness prediction", // Main title

    // États du backend
    backendStatusTitle: "Backend Status",
    statusChecking: "Checking...",
    statusOnline: "Online",
    statusOffline: "Offline",

    // Informations du modèle
    modelInfoTitle: "Model Information",
    modelInfoLoading: "Loading information...",
    modelName: "Name",
    modelVersion: "Version",
    modelImageSize: "Image Size",
    modelFormat: "Format",
    modelType: "Type",
    modelDevice: "Device",
    modelClasses: "Classes",
    modelLoaded: "Model Loaded",
    modelMode: "Mode",
    modelModeTest: "🧪 Test (mock predictions)",

    // Prédiction
    predictionTitle: "Make a prediction",
    predictionSubtitle: "Upload an image to get a prediction",
    predictionAnalyzing: "Analyzing...",
    predictionPredict: "Predict",
    predictionNewImage: "New image",
    predictionCancel: "Cancel",
    predictionOr: "or",
    predictionImageAlt: "Image of the banana for prediction",
    predictionError: "Error during prediction",

    // Caméra - nouvelles traductions
    cameraStartButton: "Start camera",
    cameraCaptureButton: "Capture",
    cameraNewPhotoButton: "New photo",
    cameraVideoLabel: "Video recording",
    cameraPermissionError: "Camera permission error",
    cameraPermissionDenied: "Camera permission denied",
    cameraNotFound: "Camera not found",
    cameraNotSupported: "Camera not supported",
    cameraInUse: "Camera in use",

    // Correction - nouvelles traductions
    correctionTitle: "Submit a correction",
    correctionSubmit: "Submit",
    correctionReset: "Start over",
    correctionBananaQuestion: "Is the banana prediction correct?",
    correctionYesBanana: "Yes, it's a banana",
    correctionNoBanana: "No, it's not a banana",
    correctionDurationTitle: "Storage duration",
    correctionDurationSubtitle: "How long has the banana been stored?",
    correctionDays: "days",
    correctionDay: "day",
    correctionStateImmediate: "Immediate state",
    correctionStateOptimal: "Optimal state",
    correctionStateConservation: "Conservation state",
    correctionStatePredicted: "Predicted state",
    correctionSendButton: "Send correction",
    correctionSending: "Sending correction...",
    correctionSuccessTitle: "Correction submitted",
    correctionSuccessMessage: "Thank you for your contribution!",
    correctionNewPredictionMessage: "A new prediction will be made.",
    correctionCurrentPrediction: "Current prediction: {}",

    // États de maturité détaillés
    stateVeryRipe: "Very ripe",
    stateRipe: "Ripe",
    stateGoodRipeness: "Good ripeness",
    stateMediumRipeness: "Medium ripeness",
    stateStillGreen: "Still green",

    // Résultats
    resultTitle: "Prediction Result",
    resultConfidence: "confidence",
    resultTopPredictions: "Top predictions",
    resultImageKey: "Image key",

    // États généraux
    yes: "Yes",
    no: "No",
    loading: "Loading",
    error: "Error",
    success: "Success",

    // Messages d'erreur
    errorBackendUnavailable: "Backend unavailable",
    errorModelInfoLoading: "Error loading model information",
    errorPrediction: "Error during prediction",

    // Classes de bananes
    classUnripe: "Unripe",
    classRipe: "Ripe",
    classOverripe: "Overripe",
    classRotten: "Rotten",
    classUnknowns: "Unknown",

    // Thème
    themeLight: "Light",
    themeDark: "Dark",
    themeSystem: "System",
    themeToggle: "Toggle theme",

    // Langue
    languageSwitch: "Switch language",
    french: "Français",
    english: "English",
  }
};

export const useTranslation = (language: Language) => {
  return translations[language];
};

export const translateClass = (className: string, language: Language): string => {
  const t = translations[language];
  switch (className.toLowerCase()) {
    case 'unripe': return t.classUnripe;
    case 'ripe': return t.classRipe;
    case 'overripe': return t.classOverripe;
    case 'rotten': return t.classRotten;
    case 'unknowns': return t.classUnknowns;
    default: return className;
  }
};
