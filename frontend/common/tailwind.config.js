/** @type {import('tailwindcss').Config} */
module.exports = {
  presets: [require('./tailwind.preset')],
  content: [
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  safelist: ['dark'],
  theme: {
    extend: {},
  },
  plugins: [],
}