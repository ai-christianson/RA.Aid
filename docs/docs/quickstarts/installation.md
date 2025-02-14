import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Installation

Create a new Python 3.12 virtual environment and install RA.Aid:

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

Once installed, see the [Recommended Configuration](recommended) to set up RA.Aid with the recommended settings.
