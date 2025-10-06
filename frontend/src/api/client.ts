import { ApiError } from '../types';
import { API_CONFIG } from '../config/api';

class ApiClient {
  private async fetchWithTimeout(
    url: string,
    options: RequestInit,
    timeout: number,
    signal?: AbortSignal
  ): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    const combinedSignal = signal
      ? this.combineAbortSignals([signal, controller.signal])
      : controller.signal;

    try {
      const response = await fetch(url, {
        ...options,
        signal: combinedSignal,
      });
      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  }

  private combineAbortSignals(signals: AbortSignal[]): AbortSignal {
    const controller = new AbortController();
    for (const signal of signals) {
      if (signal.aborted) {
        controller.abort();
        return controller.signal;
      }
      signal.addEventListener('abort', () => controller.abort(), { once: true });
    }
    return controller.signal;
  }

  async request<T>(
    url: string,
    options: RequestInit = {},
    retries = API_CONFIG.MAX_RETRIES,
    signal?: AbortSignal
  ): Promise<T> {
    const attempt = async (retriesLeft: number): Promise<T> => {
      try {
        const response = await this.fetchWithTimeout(
          url,
          {
            ...options,
            mode: 'cors',
            headers: {
              ...options.headers,
            },
          },
          API_CONFIG.TIMEOUT,
          signal
        );

        if (!response.ok) {
          const errorText = await response.text().catch(() => 'Erreur inconnue');
          let errorMessage = `Erreur ${response.status}`;

          try {
            const errorJson = JSON.parse(errorText);
            errorMessage = errorJson.detail || errorJson.message || errorMessage;
          } catch {
            errorMessage = errorText || errorMessage;
          }

          const error: ApiError = {
            message: errorMessage,
            status: response.status,
            details: errorText,
          };

          if (response.status >= 500 && retriesLeft > 0) {
            await new Promise((resolve) => setTimeout(resolve, API_CONFIG.RETRY_DELAY));
            return attempt(retriesLeft - 1);
          }

          throw error;
        }

        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          return await response.json();
        }

        return (await response.text()) as T;
      } catch (error) {
        if (error instanceof Error) {
          if (error.name === 'AbortError') {
            throw {
              message: 'Requête annulée',
              details: 'La requête a été annulée par l\'utilisateur',
            } as ApiError;
          }

          if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            if (retriesLeft > 0) {
              await new Promise((resolve) => setTimeout(resolve, API_CONFIG.RETRY_DELAY));
              return attempt(retriesLeft - 1);
            }
            throw {
              message: 'Erreur de connexion',
              details: 'Impossible de joindre le serveur. Vérifiez votre connexion.',
            } as ApiError;
          }
        }

        if ((error as ApiError).status) {
          throw error;
        }

        throw {
          message: 'Erreur inattendue',
          details: error instanceof Error ? error.message : String(error),
        } as ApiError;
      }
    };

    return attempt(retries);
  }

  async get<T>(url: string, signal?: AbortSignal): Promise<T> {
    return this.request<T>(url, { method: 'GET' }, API_CONFIG.MAX_RETRIES, signal);
  }

  async post<T>(url: string, body: unknown, signal?: AbortSignal): Promise<T> {
    return this.request<T>(
      url,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      },
      API_CONFIG.MAX_RETRIES,
      signal
    );
  }

  async postFormData<T>(url: string, formData: FormData, signal?: AbortSignal): Promise<T> {
    return this.request<T>(
      url,
      {
        method: 'POST',
        body: formData,
      },
      API_CONFIG.MAX_RETRIES,
      signal
    );
  }
}

export const apiClient = new ApiClient();
