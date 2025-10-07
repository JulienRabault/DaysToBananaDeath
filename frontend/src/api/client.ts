import { API_CONFIG, API_BASE_URL } from '../config/api';
import { parseApiError } from '../utils/errorUtils';

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
    endpoint: string,
    options: RequestInit = {},
    retries = API_CONFIG.MAX_RETRIES,
    signal?: AbortSignal
  ): Promise<T> {
    // Construire l'URL complète
    const url = endpoint.startsWith('http') ? endpoint : `${API_BASE_URL}${endpoint}`;

    const startTime = Date.now();
    console.log(`[API] DÉBUT ${options.method || 'GET'} ${url}`);

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

        const duration = Date.now() - startTime;

        if (!response.ok) {
          const errorText = await response.text().catch(() => 'Erreur inconnue');
          console.error(`[API] ERREUR ${response.status} ${url} (${duration}ms)`);
          console.error('Détails:', errorText);

          let errorDetails = errorText;
          try {
            const errorJson = JSON.parse(errorText);
            errorDetails = errorJson;
          } catch {
            // Garder le texte original si ce n'est pas du JSON
          }

          // Utiliser le nouveau parser d'erreurs
          const apiError = parseApiError({
            status: response.status,
            details: errorDetails,
            message: `HTTP ${response.status}`
          });

          throw apiError;
        }

        // Succès
        console.log(`[API] SUCCÈS ${response.status} ${url} (${duration}ms)`);

        const contentType = response.headers.get('content-type');
        if (contentType?.includes('application/json')) {
          const data = await response.json();
          console.log('[API] Réponse JSON:', data);
          return data;
        } else {
          const text = await response.text();
          console.log('[API] Réponse texte:', text);
          return text as unknown as T;
        }
      } catch (error: any) {
        const duration = Date.now() - startTime;

        // Si c'est déjà une ApiError parsée, la relancer
        if (error.message && error.code) {
          throw error;
        }

        // Parser les autres types d'erreurs
        const apiError = parseApiError(error);
        console.error(`[API] ERREUR ${url} (${duration}ms):`, apiError);

        // Retry pour certaines erreurs
        if (retriesLeft > 0 && this.shouldRetry(apiError)) {
          console.log(`[API] Nouvelle tentative dans 1s... (${retriesLeft} restantes)`);
          await new Promise(resolve => setTimeout(resolve, 1000));
          return attempt(retriesLeft - 1);
        }

        throw apiError;
      }
    };

    return attempt(retries);
  }

  private shouldRetry(error: any): boolean {
    // Retry sur erreurs réseau, timeout, et certaines erreurs serveur
    const retryableCodes = ['NETWORK_ERROR', 'TIMEOUT_ERROR', 'HTTP_500', 'HTTP_502', 'HTTP_503'];
    return retryableCodes.includes(error.code);
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
