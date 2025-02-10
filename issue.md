# LLM Tool Call Fallback Feature

## Overview
Add functionality to automatically fallback to alternative LLM models when a tool call experiences multiple consecutive failures.

## Background
Currently, when a tool call fails due to LLM-related errors (e.g., API timeouts, rate limits, context length issues), there is no automatic fallback mechanism. This can lead to interrupted workflows and poor user experience.

## Relevant Files
- ra_aid/agents/ciayn_agent.py
- ra_aid/llm.py
- ra_aid/agent_utils.py
- ra_aid/__main__.py
- ra_aid/models_params.py


## Implementation Details

### Configuration
- Add new configuration value `max_tool_failures` (default: 3) to track consecutive failures before triggering fallback
- Add new command line argument `--no-fallback-tool` to disable fallback behavior (enabled by default)
- **Add new command line argument** `--fallback-tool-models` to specify a comma-separated list of fallback tool models (default: "gpt-3.5-turbo,gpt-4")  
  This list defines the fallback model sequence used by forced tool calls (via `bind_tools`) when tool call failures occur.
- Track failure count per tool call context
- Reset failure counter on successful tool call
- Store fallback model sequence per provider
- Need to validate if ENV vars are set for provider usage of that fallback model
  before usage, if that fallback ENV is not available then fallback to the next model
- Have default list of common models, first try `claude-3-5-sonnet-20241022` but
  have many alternative fallback models.

### Tool Call Wrapper
Create a new wrapper function to handle tool call execution with fallback logic:

```python
def execute_tool_with_fallback(tool_call_func, *args, **kwargs):
    failures = 0
    max_failures = get_config().max_tool_failures

    while failures < max_failures:
        try:
            return tool_call_func(*args, **kwargs)
        except LLMError as e:
            failures += 1
            if failures >= max_failures:
                # Use forced tool call via bind_tools with retry:
                llm_retry = llm_model.with_retry(stop_after_attempt=3)  # Try three times
                try_fallback_model(force=True, model=llm_retry)
                # Merge fallback model chat messages back into the original chat history.
                merge_fallback_chat_history()
                failures = 0  # Reset counter for new model
            else:
                raise
```

The prompt passed to `try_fallback_model`, should be the failed last few failing tool calls.

### Model Fallback Sequence
Define fallback sequences for each provider based on model capabilities:

1. Try same provider's smaller models
2. Try alternative providers' equivalent models
3. Raise final error if all fallbacks fail

### Provider Strategy Updates
Update provider strategies to support fallback configuration:
- Add provider-specific fallback sequences
- Handle model capability validation during fallback
- Track successful/failed attempts

## Risks and Mitigations
1. **Performance Impact**
   - Risk: Multiple fallback attempts could increase latency
   - Mitigation: Set reasonable max_failures limit and timeouts

2. **Consistency**
   - Risk: Different models may give slightly different outputs
   - Mitigation: Validate output schema consistency across models

3. **Cost**
   - Risk: Fallback to more expensive models
   - Mitigation: Configure cost limits and preferred fallback sequences

4. **State Management** 
   - Risk: Loss of context during fallbacks
   - Mitigation: Preserve conversation state and tool context

## Acceptance Criteria
1. Tool calls automatically attempt fallback models after N consecutive failures
2. `--no-fallback-tool` argument successfully disables fallback behavior
3. Fallback sequence respects provider and model capabilities
4. Original error is preserved if all fallbacks fail
5. Unit tests cover fallback scenarios and edge cases
6. README.md updated to reflect new behavior

## Testing
1. Unit tests for fallback wrapper
2. Integration tests with mock LLM failures 
3. Provider strategy fallback tests
4. Command line argument handling
5. Error preservation and reporting
6. Performance impact measurement
7. Edge cases (e.g., partial failures, timeout handling)
8. State preservation during fallbacks

## Documentation Updates
1. Add fallback feature to main README
2. Document `--no-fallback-tool` in CLI help
3. Document provider-specific fallback sequences

## Future Considerations
1. Allow custom fallback sequences via configuration
2. Add monitoring and alerting for fallback frequency
3. Optimize fallback selection based on historical success rates
4. Cost-aware fallback routing
