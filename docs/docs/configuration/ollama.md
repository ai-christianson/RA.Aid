# Ollama Integration

## Introduction
Ollama is a framework for running large language models (LLMs) locally on your machine. RA.Aid provides robust integration with Ollama, allowing you to leverage locally-hosted models without requiring external API access or keys.

This integration gives you the ability to:
- Run RA.Aid entirely offline using local models
- Avoid API costs associated with cloud LLM providers
- Use custom fine-tuned models specific to your needs
- Control your data security by keeping all interactions local

## Requirements and Setup

### Installation

1. Install Ollama following the instructions on the [official website](https://ollama.com/download)
2. Verify Ollama is running with:
   ```bash
   ollama list
   ```
3. Pull the models you want to use:
   ```bash
   ollama pull justinledwards/mistral-small-3.1-Q6_K
   ollama pull qwq:32b
   ```

### Compatible Models

RA.Aid works with many Ollama models, including:

| Model Name | Description |
|------------|-------------|
| justinledwards/mistral-small-3.1-Q6_K | Mistral AI's optimized small model |
| qwq:32b | High-performing yet small reasoning model |
| MHKetbi/Qwen2.5-Coder-32B-Instruct | Qwen 2.5 Coder Instruct model |

You can also use any custom model you've created or imported into Ollama.

## Configuration Options

### Environment Variables

Ollama configuration is primarily controlled through the following environment variables:

| Environment Variable | Purpose | Default Value |
|----------------------|---------|---------------|
| `OLLAMA_BASE_URL` | The URL where Ollama is running | `http://localhost:11434` |
| `EXPERT_OLLAMA_BASE_URL` | Separate URL for expert models | Same as `OLLAMA_BASE_URL` |

Unlike other providers (OpenAI, Anthropic, etc.), Ollama doesn't require an API key since it runs locally.

### Command-Line Parameters

RA.Aid provides several command-line parameters specifically for Ollama configuration:

```bash
ra-aid --provider ollama --model <model_name> [options]
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--provider ollama` | Sets Ollama as the LLM provider | - |
| `--model <model_name>` | Specifies which Ollama model to use | - |
| `--num-ctx <number>` | Sets the context window size in tokens | 262144 |
| `--expert-provider ollama` | Uses Ollama for expert queries | - |
| `--expert-model <model_name>` | Sets which Ollama model to use for expert queries | - |
| `--expert-num-ctx <number>` | Sets the expert context window size | 262144 |
| `--temperature <float>` | Controls response randomness (0.0-1.0) | 0.7 |

## Usage Examples

### Basic Usage

Run RA.Aid with Ollama's justinledwards/mistral-small-3.1-Q6_K model:

```bash
ra-aid --provider ollama --model justinledwards/mistral-small-3.1-Q6_K -m "Add unit tests to the database module"
```

### Adjusting Context Window Size

For complex tasks that require more context:

```bash
ra-aid --provider ollama --model justinledwards/mistral-small-3.1-Q6_K --num-ctx 8192 -m "Refactor the entire error handling system"
```

### Using Different Models for Expert Mode

Configure separate models for main tasks and expert queries:

```bash
ra-aid --provider ollama --model justinledwards/mistral-small-3.1-Q6_K --expert-provider ollama --expert-model qwq:32b -m "Create a React component for user authentication"
```

## Ollama-Specific Features

### Context Window Size (`num-ctx`)

The `--num-ctx` parameter controls how many tokens the model can process at once, affecting:

1. **Input Processing**: Larger values allow RA.Aid to provide more context from your codebase
2. **Memory Capacity**: How much information the model can reference while working
3. **Generation Length**: Influences how detailed the responses can be

The optimal value depends on:
- The specific model you're using (some models have built-in limits)
- Your available system resources (larger values require more RAM)
- The complexity of your project and tasks

For most development tasks, values between 4,096 and 16,384 work well. Extremely large values (like the default 262,144) are typically unnecessary and may consume excessive resources.

### Local Model Advantages

Using Ollama with RA.Aid provides several benefits:

- **Privacy**: Your code and prompts never leave your machine
- **No Rate Limits**: Unlimited usage without API quotas or rate limits
- **Offline Operation**: Work without internet connectivity
- **Cost-Free**: No usage-based billing or API charges
- **Customization**: Use custom models fine-tuned for your specific needs

## Troubleshooting

### Common Issues

#### Ollama Not Running

**Symptoms**: RA.Aid reports connection errors when trying to use Ollama models.

**Solution**: 
1. Verify Ollama is running with `ollama list`
2. Start Ollama if needed (this varies by OS): `ollama serve`
3. Check if Ollama is running on a different port and set `OLLAMA_BASE_URL` accordingly

#### Model Not Found

**Symptoms**: Error message indicating the specified model doesn't exist.

**Solution**:
1. List available models with `ollama list`
2. Pull the model you want: `ollama pull <model_name>`
3. Check for typos in the model name

#### Out of Memory Errors

**Symptoms**: Ollama crashes or returns errors when processing large inputs.

**Solution**:
1. Use a smaller `--num-ctx` value
2. Try a more memory-efficient model
3. Close other memory-intensive applications
4. Ensure your system has enough RAM for the chosen model

#### Slow Performance

**Symptoms**: Responses take a very long time to generate.

**Solution**:
1. Use a smaller or more efficient model
2. Reduce the `--num-ctx` value
3. Check if your GPU is being utilized (if available)
4. Consider using quantized versions of models and/or smaller models

### Best Practices

1. **Match Model to Task**: Use code-specific models for programming tasks and general models for research or planning.

2. **Start with Smaller Context**: Begin with a modest `--num-ctx` value (4096-8192) and increase only if needed.

3. **Quantized Models**: For better performance on consumer hardware, use quantized models (identified by q4, q5, etc. in their name).

4. **GPU Acceleration**: If you have a compatible GPU, ensure Ollama is configured to use it for significant speed improvements.

## Expert Model Configuration

RA.Aid's expert model allows using different models for specialized query types. With Ollama, you can configure this using:

```bash
ra-aid --provider ollama --model justinledwards/mistral-small-3.1-Q6_K --expert-provider ollama --expert-model qwq:32b -m "Your task here"
```

This configuration uses:
- `justinledwards/mistral-small-3.1-Q6_K` for the main task planning and implementation
- `qwq:32b` for expert queries requiring deeper code analysis

You can also use different providers for main and expert functions:

```bash
ra-aid --provider ollama --model justinledwards/mistral-small-3.1-Q6_K --expert-provider anthropic --expert-model claude-3-sonnet-20240229 -m "Your task here"
```

This hybrid approach lets you:
- Use local models for most operations, saving API costs
- Leverage cloud models only for complex expert queries
- Optimize your workflow by matching models to specific query types

In addition to the expert model for domain questions, RA.Aid supports the `--reasoning-assistance` flag which enables the expert model (qwq:32b) to help the primary model make optimal use of available tools. This functionality is separate from the expert query system and can significantly improve the performance of smaller models by providing guidance on tool selection and usage. For complete details on this feature, see our [Reasoning Assistance](./reasoning-assistance.md) guide.