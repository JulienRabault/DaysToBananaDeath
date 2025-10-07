import { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useSettings } from '../store/useSettings';
import { applyTheme } from '../utils/theme';
import { BananaPixelIcon } from './BananaPixelIcon';

interface LayoutProps {
  children: ReactNode;
}

export const Layout = ({ children }: LayoutProps) => {
  const location = useLocation();
  const { theme, setTheme } = useSettings();

  const toggleTheme = () => {
    const newTheme = theme === 'dark' ? 'light' : theme === 'light' ? 'system' : 'dark';
    setTheme(newTheme);
    applyTheme(newTheme);
  };

  const getThemeIcon = () => {
    if (theme === 'dark') return '🌙';
    if (theme === 'light') return '☀️';
    return '💻';
  };

  const navLinks = [
    { path: '/', label: 'Accueil' },
    { path: '/predict', label: 'Prédiction' },
    { path: '/settings', label: 'Paramètres' },
  ];

  return (
    <div className="flex min-h-screen flex-col bg-banana-50 dark:bg-gray-950">
      <header className="relative border-b border-yellow-200/60 bg-white/90 backdrop-blur dark:border-yellow-500/30 dark:bg-gray-900/80">
        <div className="absolute inset-x-0 -top-24 mx-auto h-32 w-[90%] max-w-6xl bg-gradient-to-r from-yellow-200/60 via-yellow-100/40 to-yellow-200/60 blur-3xl" aria-hidden="true" />
        <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-20 items-center justify-between">
            <nav className="flex items-center gap-6" aria-label="Navigation principale">
              <Link
                to="/"
                className="flex items-center gap-3 rounded-full bg-white/70 px-4 py-2 text-xl font-bold text-gray-900 shadow-sm ring-1 ring-yellow-300/60 transition hover:-translate-y-0.5 hover:shadow-lg dark:bg-gray-900/60 dark:text-yellow-100 dark:ring-yellow-500/40"
              >
                <BananaPixelIcon size={36} />
                <span>Banana Prediction</span>
              </Link>
              <div className="hidden items-center gap-2 rounded-full bg-white/60 px-3 py-1 text-sm shadow-sm ring-1 ring-yellow-200/60 dark:bg-gray-900/60 dark:ring-yellow-500/40 sm:flex">
                {navLinks.map((link) => (
                  <Link
                    key={link.path}
                    to={link.path}
                    className={`rounded-full px-3 py-1 font-medium transition-all ${
                      location.pathname === link.path
                        ? 'bg-yellow-300 text-gray-900 shadow-sm dark:bg-yellow-500 dark:text-gray-900'
                        : 'text-gray-700 hover:bg-yellow-200/80 dark:text-yellow-100 dark:hover:bg-yellow-500/40'
                    }`}
                    aria-current={location.pathname === link.path ? 'page' : undefined}
                  >
                    {link.label}
                  </Link>
                ))}
              </div>
            </nav>
            <button
              onClick={toggleTheme}
              className="flex items-center gap-2 rounded-full bg-white/70 px-3 py-2 text-sm font-medium text-gray-700 shadow-sm transition hover:-translate-y-0.5 hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-yellow-400 focus:ring-offset-2 dark:bg-gray-900/70 dark:text-yellow-100 dark:hover:bg-gray-900"
              aria-label={`Thème actuel : ${theme}. Cliquer pour changer`}
            >
              <span className="text-xl" aria-hidden="true">{getThemeIcon()}</span>
              <span className="hidden sm:inline">Thème</span>
            </button>
          </div>
        </div>
      </header>

      <main className="relative mx-auto flex-1 w-full max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
        <div className="absolute inset-x-0 top-0 -z-10 mx-auto h-64 w-[95%] max-w-5xl rounded-[3rem] bg-gradient-to-br from-yellow-200 via-yellow-100 to-yellow-300 opacity-70 blur-3xl dark:from-yellow-500/30 dark:via-yellow-400/20 dark:to-yellow-500/20" aria-hidden="true" />
        {children}
      </main>

      <footer className="border-t border-yellow-200/80 bg-white/90 py-8 text-sm text-gray-600 backdrop-blur dark:border-yellow-500/30 dark:bg-gray-900/80 dark:text-yellow-100">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3 font-medium">
            <BananaPixelIcon size={28} className="drop-shadow" />
            <span>Prédiction &amp; correction d&apos;images de bananes</span>
          </div>
          <p className="text-xs text-gray-500 dark:text-yellow-200/80">
            Conçu pour surveiller la maturité des bananes avec une touche de soleil tropical.
          </p>
        </div>
      </footer>
    </div>
  );
};
