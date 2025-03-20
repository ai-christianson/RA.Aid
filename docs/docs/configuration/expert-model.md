# Expert Model

## Introduction

RA.Aid allows you to configure an "expert model" - an AI model you select to provide advanced reasoning capabilities for complex tasks. Think of it as a "second opinion" system that helps with particularly challenging problems when the main agent needs additional assistance.

### Expert Model vs. Reasoning Assistance

It's important to understand the distinction between two related features in RA.Aid:

- **Expert Model**: Helps solve domain-specific problems about the project the agent is working on. This is used when the agent needs help analyzing code, debugging issues, or making complex decisions about the implementation.

- **[Reasoning Assistance](./reasoning-assistance.md)**: Uses the expert model in a different way - specifically to help weaker models make better decisions about tool usage and planning. This is more about helping the agent with its process rather than with domain-specific problems.

Think of the expert model as the "domain expert" that helps with technical challenges, while reasoning assistance is where that same expert provides "meta-guidance" on how the agent should approach the task.

The expert model feature is designed to help with:
- Detailed analysis of code and complex systems
- Debugging and identifying logical errors
- Making more informed decisions
- Tackling problems that require deeper reasoning

This guide explains how to configure and use the expert model feature in your RA.Aid projects.

## Configuration Options

### Command Line Arguments

You can configure the expert model through several command line arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--expert-provider` | AI provider to use for expert queries (openai, anthropic, etc.) | Falls back to available API keys |
| `--expert-model` | Specific model to use for expert queries | Provider-dependent |
| `--expert-num-ctx` | Context window size (for Ollama) | 262144 |
| `--show-thoughts` | Display thinking output from the model | False |

Example usage:
```bash
ra-aid --expert-provider anthropic --expert-model claude-3-opus-20240229
```

### Provider Selection

If you don't specify an expert provider, RA.Aid automatically selects one based on available API keys:

1. First checks for OpenAI access
2. Then checks for DeepSeek 
3. Falls back to the main provider and model if neither is available

For OpenAI specifically, if no model is specified, it automatically selects the best available reasoning model.

### Environment Variables

The expert model can be configured through these environment variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for OpenAI models |
| `ANTHROPIC_API_KEY` | API key for Anthropic models |
| `DEEPSEEK_API_KEY` | API key for DeepSeek models |
| `EXPERT_OPENAI_API_KEY` | API key specifically for expert (OpenAI) |
| `EXPERT_ANTHROPIC_API_KEY` | API key specifically for expert (Anthropic) |
| `OLLAMA_BASE_URL` | URL for Ollama API (default: http://localhost:11434) |

The `EXPERT_` prefixed variables let you use different API keys for the expert model than for the main agent.

## How the Expert Model Feature Works

The expert model works behind the scenes to provide your agent with advanced reasoning capabilities:

1. When the agent encounters a complex problem, it gathers relevant context
2. The agent presents this context along with a question to your configured expert model
3. The expert model analyzes the context and provides detailed reasoning and advice
4. The agent incorporates this advice into its work

All of this happens automatically when needed. The agent will consult the expert model in situations such as:
- Complex debugging problems
- Architecture and design decisions
- Analysis of complex codebases
- Logical reasoning challenges

## Model Capabilities

Different models you can select have different capabilities:

### Thinking Display

When used with the `--show-thoughts` flag, some expert models can display their reasoning process:

- Models like Claude show their step-by-step thinking
- This helps you understand how the model arrived at its conclusions
- Particularly useful for debugging or learning purposes

### Deterministic Answers

Expert models are typically configured for consistency rather than creativity:
- They use lower temperature settings when available
- This produces more predictable and reliable results
- Ideal for programming and logical reasoning tasks

## Best Practices

To get the most out of the expert model feature:

3. **Choose the Right Model**:
   - Typically, it is best to use a reasoning model
   - For complex reasoning tasks, use OpenAI's advanced models or Claude 3 Opus
   - For offline work, configure Ollama with a suitable model

4. **Review Thinking Output**:
   - Enable `--show-thoughts` to see the model's reasoning process
   - Use this to understand complex solutions better

## Troubleshooting

### Common Issues

1. **"Failed to initialize expert model"**:
   - Check that the specified expert provider and expert model are correct
   - Verify that the required API keys are available in environment variables
   - Ensure the model is available through your subscription

2. **Poor quality responses**:
   - Try a different model with better reasoning capabilities
   - Make sure your question is clear and specific

3. **Slow response times**:
   - Expert models prioritize quality over speed
   - For time-sensitive tasks, consider using the main agent without expert consultation

## See Also

- [Thinking Models](./thinking-models.md) - Learn more about models that expose their reasoning process
- [Reasoning Assistance](./reasoning-assistance.md) - Configure additional reasoning support for models
- [Ollama Integration](./ollama.md) - Details on using local models with Ollama
