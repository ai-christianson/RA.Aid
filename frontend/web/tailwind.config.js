/** @type {import('tailwindcss').Config} */
module.exports = {
  presets: [require('../common/tailwind.preset')],
  content: [
    './src/**/*.{js,jsx,ts,tsx}',
    '../common/src/**/*.{js,jsx,ts,tsx}'
  ],
  theme: {
    extend: {},
  },
  plugins: [
    require('@tailwindcss/forms')
  ],
}