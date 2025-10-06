import React from 'react';
import ReactDOM from 'react-dom/client';
import { App } from './App';
import './index.css';
import { useSettings } from './store/useSettings';
import { applyTheme, initTheme } from './utils/theme';

const theme = useSettings.getState().theme;
applyTheme(theme);
initTheme();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
