export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f5f7fa',
          100: '#eaeef4',
          200: '#d0dae7',
          300: '#a8bbd3',
          400: '#7a98ba',
          500: '#5879a4',
          600: '#445f89',
          700: '#384d70',
          800: '#31425e',
          900: '#2c3950',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
