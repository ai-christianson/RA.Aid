/** @type {import('tailwindcss').Config} */
module.exports = {
  // Use the same configuration as the common package
  // This ensures consistent styling across packages
  presets: [require('../common/tailwind.config.js')],
};