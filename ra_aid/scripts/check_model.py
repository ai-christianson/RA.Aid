import argparse
import os
import sys

# This allows the script to be run from the root of the project
# and still find the ra_aid module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    import litellm
except ImportError:
    print(
        "Error: 'litellm' is not installed. Please add it to your project's dependencies."
    )
    print(
        "You might need to add 'litellm' to your 'pyproject.toml' and run 'make setup-dev'."
    )
    sys.exit(1)


def check_model_availability(model_name: str) -> bool:
    """
    Checks if a model is available and valid using litellm.

    This requires the appropriate provider API key to be set as an
    environment variable (e.g., ANTHROPIC_API_KEY for Claude models).
    """
    print(f"Checking availability for model: {model_name}...")
    try:
        # litellm.model_info will raise an exception if the model is not found
        # or if the API key is invalid.
        info = litellm.get_model_info(model=model_name)
        print(f"✅ Success! Model '{model_name}' is available.")
        print(f"   Provider: {info.get('litellm_provider')}")
        print(
            f"   Input Cost (per 1M tokens): ${info.get('input_cost_per_token', 0) * 1_000_000 if info.get('input_cost_per_token') else 'N/A'}"
        )
        print(
            f"   Output Cost (per 1M tokens): ${info.get('output_cost_per_token', 0) * 1_000_000 if info.get('output_cost_per_token') else 'N/A'}"
        )
        print(f"   Context Window: {info.get('max_input_tokens')}")
        return True
    except Exception as e:
        print(f"❌ Failed to validate model '{model_name}'.")
        # The error from litellm is usually very descriptive.
        print(f"   Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Check model availability using litellm. Requires provider API keys in the environment.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="claude-3-sonnet-20240229",
        help="The name of the model to check (e.g., 'claude-3-opus-20240229', 'gpt-4o').\n"
        "Defaults to 'claude-3-sonnet-20240229'.",
    )
    args = parser.parse_args()

    # A simple way to check for any API key. This is not exhaustive.
    api_key_present = any(
        key in os.environ
        for key in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "DEEPSEEK_API_KEY",
            "GROQ_API_KEY",
            "FIREWORKS_API_KEY",
            "OPENROUTER_API_KEY",
            "REQUESTY_API_KEY",
            "MAKEHUB_API_KEY",
        ]
    )

    if (
        not api_key_present
        and not os.getenv("AWS_PROFILE")
        and not os.getenv("OLLAMA_BASE_URL")
    ):
        print(
            "Warning: No common provider API key (e.g., ANTHROPIC_API_KEY) or AWS/Ollama config found in environment."
        )
        print("The check might fail if the model requires authentication.")

    check_model_availability(args.model)


if __name__ == "__main__":
    main()
