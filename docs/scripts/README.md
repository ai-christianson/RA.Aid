# Docusaurus Scripts

This directory contains utility scripts for the Docusaurus documentation site.

## version.js

This script reads the version from `../../ra_aid/__version__.py` and creates a `version.json` file in the Docusaurus `static/` directory, which will be included in the built site.

### Usage

The script is automatically run as part of the build and start processes via npm scripts defined in `package.json`, but can also be run manually:

```bash
# From docs directory
npm run version-json

# Or directly
node scripts/version.js
```

### Output

The script creates a `static/version.json` file with the following format:

```json
{
  "version": "x.y.z"
}
```

This file will be available at `/version.json` in the built site, allowing client-side version checks.