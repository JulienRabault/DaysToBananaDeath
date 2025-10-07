import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { DEFAULT_ENDPOINTS, DEFAULT_RESPONSE_MAPPING } from '../config/api';

type Language = 'fr' | 'en';

interface Settings {
  baseUrl: string;
  endpoints: {
    health: string;
    predict: string;
    corrections: string;
  };
  responseMapping: {
    image_key: string;
    predicted_label: string;
    predicted_index: string;
    confidence: string;
    predictions: string;
    latency_ms: string;
  };
  theme: 'light' | 'dark' | 'system';
  language: Language;
  setBaseUrl: (url: string) => void;
  setEndpoint: (key: keyof Settings['endpoints'], value: string) => void;
  setResponseMapping: (key: keyof Settings['responseMapping'], value: string) => void;
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
  setLanguage: (language: Language) => void;
  resetToDefaults: () => void;
}

const getInitialBaseUrl = () => {
  return import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
};

export const useSettings = create<Settings>()(
  persist(
    (set) => ({
      baseUrl: getInitialBaseUrl(),
      endpoints: {
        health: DEFAULT_ENDPOINTS.HEALTH,
        predict: DEFAULT_ENDPOINTS.PREDICT,
        corrections: DEFAULT_ENDPOINTS.CORRECTIONS,
      },
      responseMapping: DEFAULT_RESPONSE_MAPPING,
      theme: 'system',
      language: 'en', // Changé de 'fr' à 'en' pour l'anglais par défaut
      setBaseUrl: (url) => set({ baseUrl: url.trim() }),
      setEndpoint: (key, value) =>
        set((state) => ({
          endpoints: { ...state.endpoints, [key]: value.trim() },
        })),
      setResponseMapping: (key, value) =>
        set((state) => ({
          responseMapping: { ...state.responseMapping, [key]: value.trim() },
        })),
      setTheme: (theme) => set({ theme }),
      setLanguage: (language) => set({ language }),
      resetToDefaults: () =>
        set({
          baseUrl: getInitialBaseUrl(),
          endpoints: {
            health: DEFAULT_ENDPOINTS.HEALTH,
            predict: DEFAULT_ENDPOINTS.PREDICT,
            corrections: DEFAULT_ENDPOINTS.CORRECTIONS,
          },
          responseMapping: DEFAULT_RESPONSE_MAPPING,
          theme: 'system',
          language: 'en', // Changé de 'fr' à 'en' pour l'anglais par défaut
        }),
    }),
    {
      name: 'app-settings',
      version: 2,
    }
  )
);
