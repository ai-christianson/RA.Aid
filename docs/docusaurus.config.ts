import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'RA-Aid Documentation',
  favicon: 'img/favicon.ico',
  url: 'https://docs.ra-aid.ai',
  baseUrl: '/',
  
  onDuplicateRoutes: 'ignore',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  plugins: [],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: '/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    algolia: {
      appId: 'HZFTKC6J5D',          // Optional: required if your DocSearch config uses it
      apiKey: '8e9db8266e2edda55250eada5c86d4a4',   // Public API key; safe to commit
      indexName: 'ra-aid',    // The index provided by Algolia
      contextualSearch: true,          // Optional: adjusts search results by language/version
      // Optional parameters:
      // searchParameters: {},
      // searchPagePath: 'search',
    },
    navbar: {
      logo: {
        alt: 'Site Logo',
        src: 'img/logo-black-transparent.png',
        srcDark: 'img/logo-white-transparent.gif',
        href: 'https://ra-aid.ai'
      },
      items: [
        {
          type: 'doc',
          position: 'left',
          docId: 'intro',
          label: 'Docs',
        },
        {
          href: 'https://github.com/ai-christianson/RA.Aid',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      copyright: `Copyright © ${new Date().getFullYear()} AI Christianson. Built with RA.Aid and Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: false,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
