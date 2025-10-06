import { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { useSettings } from '../store/useSettings';
import { applyTheme } from '../utils/theme';

interface LayoutProps {
  children: ReactNode;
}

export const Layout = ({ children }: LayoutProps) => {
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

  return (
    <div className="flex min-h-screen flex-col bg-gray-50 dark:bg-gray-900">
      <header className="border-b border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            <nav className="flex items-center gap-8" aria-label="Navigation principale">
              <Link
                to="/"
                className="text-xl font-bold text-gray-900 dark:text-white"
              >
                🍌 Banana Prediction
              </Link>
            </nav>
            <button
              onClick={toggleTheme}
              className="rounded-lg p-2 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 dark:hover:bg-gray-700"
              aria-label={`Thème actuel : ${theme}. Cliquer pour changer`}
            >
              <span className="text-xl" aria-hidden="true">{getThemeIcon()}</span>
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto flex-1 max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {children}
      </main>

      <footer className="border-t border-gray-200 bg-white py-6 dark:border-gray-700 dark:bg-gray-800">
        <div className="mx-auto max-w-7xl px-4 text-center text-sm text-gray-500 dark:text-gray-400">
          Banana Prediction App - Prédiction et correction d&apos;images
        </div>
      </footer>
    </div>
  );
};
