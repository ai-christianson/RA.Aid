import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Open Models Configuration

RA.Aid supports various open source model providers and configurations. This guide shows you how to configure and use different open models with RA.Aid.

## Supported Providers

<Tabs groupId="model-provider">
  <TabItem value="deepseek" label="DeepSeek" default>

### DeepSeek Models

To use DeepSeek models, you'll need a DeepSeek API key. Set it in your environment:

```bash
export DEEPSEEK_API_KEY=your_api_key_here
```

Then run RA.Aid with the deepseek provider and model:

```bash
ra-aid -m "Your task" --provider deepseek --model deepseek-reasoner
```

You can also access DeepSeek models through OpenRouter:

```bash
ra-aid -m "Your task" --provider openrouter --model deepseek/deepseek-r1
```

  </TabItem>
  <TabItem value="openrouter" label="OpenRouter">

### OpenRouter Integration

OpenRouter provides access to various open source models. First, set your API key:

```bash
export OPENROUTER_API_KEY=your_api_key_here
```

Example using Mistral:

```bash
ra-aid -m "Your task" --provider openrouter --model mistralai/mistral-large-2411
```

  </TabItem>
  <TabItem value="expert" label="Expert Configuration">

### Expert Tool Configuration 

The expert tool can be configured to use open models for complex logic and debugging tasks:

```bash
# Use DeepSeek for expert tool
export EXPERT_DEEPSEEK_API_KEY=your_deepseek_api_key
ra-aid -m "Your task" --expert-provider deepseek --expert-model deepseek-reasoner

# Use OpenRouter for expert
export EXPERT_OPENROUTER_API_KEY=your_openrouter_api_key
ra-aid -m "Your task" --expert-provider openrouter --expert-model mistralai/mistral-large-2411
```

  </TabItem>
</Tabs>

## Environment Variables

Here are all the environment variables supported for open model configuration:

- `OPENROUTER_API_KEY`: Required for OpenRouter provider
- `DEEPSEEK_API_KEY`: Required for DeepSeek provider
- `EXPERT_OPENROUTER_API_KEY`: API key for expert tool using OpenRouter provider
- `EXPERT_DEEPSEEK_API_KEY`: API key for expert tool using DeepSeek provider

## Notes and Best Practices

- Set environment variables in your shell's configuration file (e.g., `~/.bashrc` or `~/.zshrc`) for persistence
- Consider using different models for different types of tasks (e.g., DeepSeek for reasoning, Mistral for general tasks)
- Review model performance and adjust based on your specific needs
- Keep your API keys secure and never commit them to version control
