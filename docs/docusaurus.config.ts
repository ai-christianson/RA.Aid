import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from "@docusaurus/preset-classic";
import type * as Plugin from "@docusaurus/types/src/plugin";
import type * as OpenApiPlugin from "docusaurus-plugin-openapi-docs";

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

  plugins: [
    [
      'docusaurus-plugin-openapi-docs',
      {
        id: "api",
        docsPluginId: "classic",
        config: {
         api: {  // Make sure this matches the sidebar ID
            specPath: "./ra-aid.openapi.yml", // Ensure correct path
            outputDir: "docs/api", // This must match sidebars.ts
            sidebarOptions: {
              groupPathsBy: "tag",
            },
          }
        }
      },
    ]
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: '/',
          docItemComponent: '@theme/ApiItem',
          docRootComponent: "@theme/DocRoot",
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themes: ['docusaurus-theme-openapi-docs'],

  themeConfig: {
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
