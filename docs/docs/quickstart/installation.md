---
sidebar_position: 1
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Installation

Getting started is easy! You can install RA.Aid using any of these three methods - pick the one that works best for you:

<Tabs groupId="install-method">
  <TabItem value="uv" label="UV" default>

Create a new Python 3.12 virtual environment and install RA.Aid:

First install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv venv -p 3.12
```

<Tabs groupId="operating-system">
  <TabItem value="unix" label="Unix/macOS">

```bash
source .venv/bin/activate
```

  </TabItem>
  <TabItem value="windows" label="Windows">

```bash
.venv\Scripts\activate
```

  </TabItem>
</Tabs>

```bash
uv pip install ra-aid
```

  </TabItem>
  <TabItem value="pip" label="pip">

Install RA.Aid using pip:

```bash
pip install ra-aid
```

:::note
If you're using Python 3.13 or newer, we recommend using the UV installation method instead due to compatibility issues with newer Python versions.
:::

  </TabItem>
  <TabItem value="macos" label="macOS">

Install RA.Aid using Homebrew:

```bash
brew tap ai-christianson/homebrew-ra-aid
brew install ra-aid
```

  </TabItem>
</Tabs>

Once installed, see the [Recommended Configuration](recommended) to set up RA.Aid with the recommended settings.
