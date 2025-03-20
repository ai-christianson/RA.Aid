# Reasoning Assistance

## Overview

Reasoning Assistance is a feature in RA.Aid that helps weaker models make better decisions about tool usage and task planning. It leverages a stronger model (typically your expert model) to provide strategic guidance to the main agent model at the beginning of each agent stage.

### Reasoning Assistance vs. Expert Model

It's important to understand the distinction between two related features in RA.Aid:

- **Reasoning Assistance**: Uses the expert model to help weaker agent models with their process - specifically choosing the right tools and planning their approach. This is about improving how the agent works.

- **[Expert Model](./expert-model.md)**: Helps solve domain-specific problems about the project the agent is working on. When the agent encounters technical challenges like debugging or complex implementation decisions, it can consult the expert model directly.

Think of reasoning assistance as having a mentor who guides your process and methodology, while the expert model is like consulting a domain specialist for specific technical challenges.

This feature is particularly useful when working with less capable models that may struggle with complex reasoning, tool selection, or planning. By providing expert guidance upfront, these models can perform more effectively and produce better results.

## How It Works

When reasoning assistance is enabled, RA.Aid performs the following steps at the beginning of each agent stage (research, planning, implementation):

1. Makes a one-off call to the expert model with a specialized prompt that includes:
   - A description of the current task and stage
   - The complete list of available tools
   - Instructions to provide strategic guidance on approaching the task

2. Incorporates the expert model's response into the main agent's prompt.

3. The main agent then proceeds with execution, guided by the expert's recommendations on which tools to use and how to approach the task

## Configuration

### Command Line Flags

You can enable or disable reasoning assistance using these command-line flags:

```bash
# Enable reasoning assistance
ra-aid -m "Your task description" --reasoning-assistance

# Disable reasoning assistance (overrides model defaults)
ra-aid -m "Your task description" --no-reasoning-assistance
```

## Examples

### Using Reasoning Assistance with Weaker Models

```bash
# Use qwen-qwq-32b as the expert model to provide guidance
ra-aid --model qwen-32b-coder-instruct --expert-model qwen-qwq-32b --reasoning-assistance -m "Create a simple web server in Python" 
```

### Disabling Reasoning Assistance for Strong Models

Reasoning assistance has different defaults depending on which model is used. If you would like to explicitly disable reasoning assistance, use the `--no-reasoning-assistance` flag.

```bash
# Use Claude 3 Opus without reasoning assistance
ra-aid -m "Create a simple web server in Python" --model claude-3-opus-20240229 --no-reasoning-assistance
```

## Benefits and Use Cases

Reasoning assistance provides several advantages:

1. **Better Tool Selection**: Helps models choose the right tools for specific tasks
2. **Improved Planning**: Provides strategic guidance on how to approach complex problems
3. **Reduced Errors**: Decreases the likelihood of tool misuse or inefficient approaches
4. **Model Flexibility**: Allows using weaker models more effectively by augmenting their reasoning capabilities
5. **Consistency**: Ensures more consistent behavior across different models

Common use cases include:

- Working with open-source models that have less robust tool use capabilities
- Tackling complex tasks that require careful planning and tool sequencing
- Ensuring consistent behavior when switching between different models

## Best Practices

For optimal results with reasoning assistance:

1. **Use Strong Expert Models**: The quality of reasoning assistance depends on the expert model's capabilities. Use the strongest model available for the expert role.

2. **Enable for Weaker Models**: Enable reasoning assistance by default for models known to struggle with tool selection or complex reasoning.

3. **Disable for Strong Models**: Models like Claude 3 Opus or GPT-4 typically don't need reasoning assistance and might perform better without it.

4. **Custom Tasks**: For highly specialized or unusual tasks, manually enabling reasoning assistance can be beneficial even for stronger models.

5. **Review Generated Guidance**: If debugging issues, examine the expert guidance provided to understand how it's influencing the agent's behavior.

## Troubleshooting

Common issues and solutions:

| Issue | Possible Solution |
|-------|-------------------|
| Reasoning assistance seems to make no difference | Verify both `--reasoning-assistance` flag is set and check the logs to confirm the expert model is being called |
| Expert model provides irrelevant or incorrect agent guidance | Try using a stronger expert model with `--expert-model` flag |
| Agent ignores expert guidance | Some models may not correctly follow the guidance format; try a different agent model |
| Slow performance | Reasoning assistance requires an additional model call at the start of each stage; disable it for simpler tasks if speed is critical |
| Conflicting approach with custom instructions | If you're providing specific instructions that conflict with reasoning assistance, use `--no-reasoning-assistance` |

If problems persist, check if the expert model and agent model are compatible, and consider adjusting the temperature setting to control randomness in both models.

## See Also

- [Expert Model Configuration](./expert-model.md) - Learn about the expert model configuration used for reasoning assistance
- [Thinking Models](./thinking-models.md) - Information about models that can reveal their reasoning process