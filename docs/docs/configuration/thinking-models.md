# Thinking Models

RA.Aid supports models that can reveal their internal reasoning process, providing greater transparency into how they arrive at their responses. This feature, called "thinking models," helps users better understand the model's decision-making and logic.

## Overview

Thinking models allow you to see the model's internal reasoning process separately from its final response. This offers several benefits:

- **Transparency**: Understand how the model interprets your instructions and reasons through problems
- **Debugging**: Identify where a model's reasoning might go astray
- **Learning**: Gain insights into the model's approach to problem-solving
- **Trust**: Build greater confidence in the model's outputs by seeing its thought process

RA.Aid extracts and displays thinking content in special "💭 Thoughts" panels, keeping the main response clean while still providing access to the reasoning behind it.

## How Thinking Models Work in RA.Aid

RA.Aid supports two different methods for implementing thinking models:

### 1. Explicit Think Tags

Some models, like `qwen-qwq-32b`, use explicit XML-style thinking tags to delineate their reasoning process:

```
<think>
First, I need to understand what this code does.
The function seems to be parsing a configuration file...
</think>

The function parse_config() has an issue with its error handling...
```

RA.Aid extracts the content between these `<think>...</think>` tags and displays it separately from the main response.

### 2. Native Thinking Mode

More advanced models, like Claude 3.7 Sonnet, have native thinking capabilities built in at the API level. When RA.Aid uses these models, it sends special configuration parameters in the API request:

```python
{"thinking": {"type": "enabled", "budget_tokens": 12000}}
```

These models return structured responses with separate thinking and response content, which RA.Aid processes and displays accordingly.

## Configuration and Setup

### Enabling Thinking Models

To enable the display of thinking content, use the `--show-thoughts` CLI flag when running RA.Aid:

```bash
ra-aid -m "Add error handling to the database module" --show-thoughts
```

When this flag is enabled, RA.Aid will display thinking content in separate panels whenever it's available from the model.

### Supported Models

Currently, the following models support thinking mode in RA.Aid:

| Model | Provider | Type |
|-------|----------|------|
| qwen-qwq-32b | openai-compatible | Explicit think tags |
| claude-3-7-sonnet-20250219 | anthropic | Native thinking mode |

Each model's support is configured in the `models_params.py` file using the appropriate parameter.

## Examples and Usage

### Using a Model with Explicit Think Tags

When using the `qwen-qwq-32b` model with the `--show-thoughts` flag:

```bash
ra-aid -m "Refactor the error handling logic" --provider openai-compatible --model qwen-qwq-32b --show-thoughts
```

The model might include explicit think tags in its response:

```
<think>
Let me analyze the existing error handling logic:
1. Current approach uses try/except blocks scattered throughout
2. Error messages are inconsistent
3. There's no central logging mechanism
I should suggest a unified error handling approach with proper logging.
</think>

I recommend refactoring the error handling logic by implementing a centralized error handler...
```

RA.Aid will extract this thinking content and display it in a separate panel titled "💭 Thoughts", while showing only the actual response in the main output.

### Using a Model with Native Thinking

When using Claude 3.7 Sonnet with the `--show-thoughts` flag:

```bash
ra-aid -m "Debug the database connection issue" --provider anthropic --model claude-3-7-sonnet-20250219 --show-thoughts
```

RA.Aid configures the model to use its native thinking mode, and then processes the structured response to show thinking content separately.

### Without the --show-thoughts Flag

If you run RA.Aid without the `--show-thoughts` flag, the thinking content is still extracted from the model responses, but it won't be displayed in the console. This gives you a cleaner output focused only on the model's final responses.

## Troubleshooting and Best Practices

### Common Issues

#### Thinking content not appearing

If you're not seeing thinking content despite using the `--show-thoughts` flag:

- Ensure you're using a model that supports thinking (qwen-qwq-32b or claude-3-7-sonnet-20250219)
- Verify that the model is properly configured in your environment
- Check that the model is actually including thinking content in its responses (not all prompts will generate thinking)

#### Excessive or irrelevant thinking

If the thinking content is too verbose or irrelevant:

- Try to formulate more specific and concise prompts
- Consider using a different model if the thinking style doesn't meet your needs

### Best Practices

For the most effective use of thinking models:

1. **Use selectively**: Enable `--show-thoughts` when you need to understand the model's reasoning process, but consider disabling it for routine tasks to keep output concise.

2. **Choose the right model**: Different models have different thinking styles. Claude models tend to provide more structured and methodical reasoning, while other models might have different approaches.

3. **Ask questions that benefit from reasoning**: Complex problem-solving, debugging, and analysis tasks benefit most from seeing the model's thought process.

4. **Compare thinking with output**: Use the thinking content to evaluate the quality of the model's reasoning and identify potential flaws in its approach.

5. **Provide clear instructions**: When the model's thinking seems off-track, provide clearer instructions in your next prompt to guide its reasoning process.

