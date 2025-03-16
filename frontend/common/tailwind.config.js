/** @type {import('tailwindcss').Config} */
module.exports = {
  presets: [require('./tailwind.preset')],
  content: [
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  safelist: [
    'dark',
    {
      pattern: /^dark:/,
      variants: ['hover', 'focus', 'active']
    }
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}