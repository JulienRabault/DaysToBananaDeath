import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { health } from '../api/endpoints';
import { ErrorAlert } from '../components/ErrorAlert';
import { Spinner } from '../components/Spinner';
import { BananaPixelIcon } from '../components/BananaPixelIcon';

export const Home = () => {
  const [status, setStatus] = useState<'loading' | 'online' | 'offline'>('loading');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await health();
        setStatus('online');
      } catch (err) {
        setStatus('offline');
        setError(err instanceof Error ? err.message : 'Backend non disponible');
      }
    };

    checkHealth();
  }, []);

  return (
    <div className="flex flex-col gap-12">
      <section className="relative overflow-hidden rounded-3xl bg-white/80 p-10 shadow-xl ring-1 ring-yellow-200/70 backdrop-blur dark:bg-gray-900/70 dark:ring-yellow-500/30">
        <div className="absolute -right-12 -top-12 h-48 w-48 rounded-full bg-gradient-to-br from-yellow-200/70 to-yellow-400/60 blur-2xl" aria-hidden="true" />
        <div className="absolute -bottom-16 -left-16 h-48 w-48 rounded-full bg-gradient-to-tr from-yellow-300/70 to-yellow-500/60 blur-2xl" aria-hidden="true" />
        <div className="relative grid gap-8 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
          <div className="space-y-6">
            <div className="inline-flex items-center gap-3 rounded-full bg-yellow-100/80 px-4 py-2 text-sm font-medium text-yellow-800 dark:bg-yellow-500/30 dark:text-yellow-100">
              <BananaPixelIcon size={32} />
              <span>Banana Prediction, le baromètre de maturité tropical</span>
            </div>
            <h1 className="text-4xl font-extrabold tracking-tight text-gray-900 sm:text-5xl dark:text-yellow-50">
              Surveillez la vie secrète de vos bananes
            </h1>
            <p className="text-lg text-gray-600 dark:text-yellow-100/90">
              Téléversez vos photos, laissez l&apos;IA analyser la maturité et profitez d&apos;une interface ensoleillée pour suivre, corriger et améliorer vos prédictions de bananes.
            </p>
            <div className="flex flex-wrap gap-4">
              <Link
                to="/predict"
                className="inline-flex items-center gap-2 rounded-full bg-yellow-400 px-6 py-3 text-base font-semibold text-gray-900 shadow-lg transition hover:-translate-y-0.5 hover:bg-yellow-300 focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:ring-offset-2 dark:bg-yellow-500 dark:hover:bg-yellow-400"
              >
                Commencer une prédiction
                <span aria-hidden="true">🍌</span>
              </Link>
              <Link
                to="/settings"
                className="inline-flex items-center gap-2 rounded-full border border-yellow-300/80 bg-white/70 px-6 py-3 text-base font-semibold text-yellow-700 transition hover:-translate-y-0.5 hover:bg-yellow-100 focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:ring-offset-2 dark:border-yellow-400/60 dark:bg-gray-900/60 dark:text-yellow-100 dark:hover:bg-gray-900"
              >
                Paramétrer l&apos;appli
                <span aria-hidden="true">⚙️</span>
              </Link>
            </div>
            <div className="flex flex-wrap items-center gap-6 text-sm text-gray-500 dark:text-yellow-100/80">
              <div className="flex items-center gap-2">
                <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-yellow-200 text-base">🌞</span>
                Interface pensée pour la clarté et l&apos;accessibilité
              </div>
              <div className="flex items-center gap-2">
                <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-yellow-200 text-base">🔄</span>
                Boucle de correction continue
              </div>
            </div>
          </div>
          <div className="relative rounded-3xl border border-yellow-200/70 bg-gradient-to-br from-yellow-100 via-yellow-200/70 to-yellow-300/80 p-8 text-gray-900 shadow-inner dark:border-yellow-500/30 dark:from-yellow-400/10 dark:via-yellow-500/10 dark:to-yellow-400/20 dark:text-yellow-50">
            <h2 className="mb-6 flex items-center gap-3 text-xl font-semibold">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-2xl bg-white text-2xl shadow-md dark:bg-gray-950">
                <BananaPixelIcon size={28} />
              </span>
              Thermomètre des bananes
            </h2>
            <p className="mb-4 text-sm text-gray-700 dark:text-yellow-100/90">
              Statut du backend FastAPI qui nourrit nos prédictions.
            </p>
            {status === 'loading' && (
              <div className="flex items-center gap-3 rounded-2xl bg-white/70 px-4 py-3 text-gray-700 shadow-sm dark:bg-gray-900/70 dark:text-yellow-100">
                <Spinner size="sm" />
                <span>Vérification en cours...</span>
              </div>
            )}

            {status === 'online' && (
              <div className="flex items-center gap-3 rounded-2xl bg-white/80 px-4 py-3 text-green-700 shadow-sm dark:bg-gray-900/70 dark:text-green-300">
                <div className="h-3 w-3 rounded-full bg-green-500" aria-hidden="true" />
                <span className="font-medium">Le service est ensoleillé</span>
              </div>
            )}

            {status === 'offline' && (
              <div className="space-y-3">
                <div className="flex items-center gap-3 rounded-2xl bg-white/90 px-4 py-3 text-red-600 shadow-sm dark:bg-gray-900/70 dark:text-red-300">
                  <div className="h-3 w-3 rounded-full bg-red-500" aria-hidden="true" />
                  <span className="font-medium">Oups, la grappe est hors ligne</span>
                </div>
                {error && <ErrorAlert error={error} />}
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="grid gap-8 lg:grid-cols-3">
        <div className="rounded-2xl border border-yellow-200/60 bg-white/80 p-8 shadow-lg transition hover:-translate-y-1 hover:shadow-xl dark:border-yellow-500/30 dark:bg-gray-900/70">
          <h3 className="mb-3 flex items-center gap-3 text-lg font-semibold text-gray-900 dark:text-yellow-50">
            <span className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-yellow-100 text-2xl dark:bg-yellow-500/30">📸</span>
            Capture express
          </h3>
          <p className="text-sm leading-6 text-gray-600 dark:text-yellow-100/90">
            Importez une image ou prenez une photo depuis votre appareil. Nous avons optimisé le flux pour qu&apos;il soit clair, rapide et adapté mobile.
          </p>
        </div>
        <div className="rounded-2xl border border-yellow-200/60 bg-white/80 p-8 shadow-lg transition hover:-translate-y-1 hover:shadow-xl dark:border-yellow-500/30 dark:bg-gray-900/70">
          <h3 className="mb-3 flex items-center gap-3 text-lg font-semibold text-gray-900 dark:text-yellow-50">
            <span className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-yellow-100 text-2xl dark:bg-yellow-500/30">🧠</span>
            Analyse guidée
          </h3>
          <p className="text-sm leading-6 text-gray-600 dark:text-yellow-100/90">
            L&apos;IA évalue le stade de maturité et vous présente un score de confiance lisible, avec une palette couleurs accessible et contrastée.
          </p>
        </div>
        <div className="rounded-2xl border border-yellow-200/60 bg-white/80 p-8 shadow-lg transition hover:-translate-y-1 hover:shadow-xl dark:border-yellow-500/30 dark:bg-gray-900/70">
          <h3 className="mb-3 flex items-center gap-3 text-lg font-semibold text-gray-900 dark:text-yellow-50">
            <span className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-yellow-100 text-2xl dark:bg-yellow-500/30">💬</span>
            Boucle d&apos;amélioration
          </h3>
          <p className="text-sm leading-6 text-gray-600 dark:text-yellow-100/90">
            Soumettez vos corrections pour renforcer le modèle et maintenir votre stock de bananes à point.
          </p>
        </div>
      </section>

      <section className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-3xl border border-yellow-200/60 bg-white/80 p-8 shadow-lg dark:border-yellow-500/30 dark:bg-gray-900/70">
          <h2 className="mb-4 text-2xl font-bold text-gray-900 dark:text-yellow-50">Pourquoi ce mini-site ?</h2>
          <p className="text-sm leading-7 text-gray-600 dark:text-yellow-100/90">
            Banana Prediction est né d&apos;un besoin tout simple : suivre la maturité des bananes pour éviter le gaspillage alimentaire et servir les meilleures bananes dans nos préparations. Ce mini-site est une petite lettre d&apos;amour aux bananes mûries à point et aux interfaces solaires.
          </p>
          <p className="mt-4 text-sm leading-7 text-gray-600 dark:text-yellow-100/90">
            Ici on raconte notre histoire, on partage les expériences et on offre une porte d&apos;entrée ludique vers les prédictions tropicales.
          </p>
        </div>
        <div className="rounded-3xl border border-yellow-200/60 bg-white/80 p-8 shadow-lg dark:border-yellow-500/30 dark:bg-gray-900/70">
          <h2 className="mb-4 text-2xl font-bold text-gray-900 dark:text-yellow-50">L&apos;humeur du verger</h2>
          <div className="space-y-5 text-sm leading-7 text-gray-600 dark:text-yellow-100/90">
            <div className="flex items-start gap-3">
              <span className="mt-1 inline-flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-2xl bg-yellow-100 text-xl dark:bg-yellow-500/30">🍯</span>
              <p>
                Des bananes flambées pour célébrer nos réussites UI et garder la flamme créative allumée.
              </p>
            </div>
            <div className="flex items-start gap-3">
              <span className="mt-1 inline-flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-2xl bg-yellow-100 text-xl dark:bg-yellow-500/30">🎨</span>
              <p>
                Une palette ensoleillée, des icônes pixel art et des micro-animations qui apportent du sourire sans le jargon des standards.
              </p>
            </div>
            <div className="flex items-start gap-3">
              <span className="mt-1 inline-flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-2xl bg-yellow-100 text-xl dark:bg-yellow-500/30">🪴</span>
              <p>
                On cultive le projet à l&apos;instinct : feedbacks, tests et une bonne dose de soleil pour que l&apos;IHM reste vivante.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
