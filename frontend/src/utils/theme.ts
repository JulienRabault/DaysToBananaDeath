export const applyTheme = (theme: 'light' | 'dark' | 'system') => {
  const root = window.document.documentElement;

  if (theme === 'system') {
    const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light';
    root.classList.toggle('dark', systemTheme === 'dark');
  } else {
    root.classList.toggle('dark', theme === 'dark');
  }
};

export const initTheme = () => {
  const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

  const handleChange = () => {
    const theme = localStorage.getItem('banana-app-settings');
    if (theme) {
      const settings = JSON.parse(theme);
      if (settings.state?.theme === 'system') {
        applyTheme('system');
      }
    }
  };

  mediaQuery.addEventListener('change', handleChange);
  return () => mediaQuery.removeEventListener('change', handleChange);
};
