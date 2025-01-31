"""
List of model parameters
"""

DEFAULT_TOKEN_LIMIT = 100000

models_params = {
    "openai": {
        "gpt-3.5-turbo-0125": {
            "token_limit": 16385,
            "supports_temperature": False
        },
        "gpt-3.5": {
            "token_limit": 4096,
            "supports_temperature": False
        },
        "gpt-3.5-turbo": {
            "token_limit": 16385,
            "supports_temperature": False
        },
        "gpt-3.5-turbo-1106": {
            "token_limit": 16385,
            "supports_temperature": False
        },
        "gpt-3.5-turbo-instruct": {
            "token_limit": 4096,
            "supports_temperature": False
        },
        "gpt-4-0125-preview": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gpt-4-turbo-preview": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gpt-4-turbo": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gpt-4-turbo-2024-04-09": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gpt-4-1106-preview": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gpt-4-vision-preview": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gpt-4": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "gpt-4-0613": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "gpt-4-32k": {
            "token_limit": 32768,
            "supports_temperature": False
        },
        "gpt-4-32k-0613": {
            "token_limit": 32768,
            "supports_temperature": False
        },
        "gpt-4o": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gpt-4o-2024-08-06": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gpt-4o-2024-05-13": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gpt-4o-mini": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "o1-preview": {
            "token_limit": 128000,
            "supports_temperature": True
        },
        "o1-mini": {
            "token_limit": 128000,
            "supports_temperature": True
        }
    },
    "azure_openai": {
        "gpt-3.5-turbo-0125": {
            "token_limit": 16385,
            "supports_temperature": False
        },
        "gpt-3.5": {
            "token_limit": 4096,
            "supports_temperature": False
        },
        "gpt-3.5-turbo": {
            "token_limit": 16385,
            "supports_temperature": False
        },
        "gpt-3.5-turbo-1106": {
            "token_limit": 16385,
            "supports_temperature": False
        },
        "gpt-3.5-turbo-instruct": {
            "token_limit": 4096,
            "supports_temperature": False
        },
        "gpt-4-0125-preview": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gpt-4-turbo-preview": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gpt-4-turbo": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gpt-4-turbo-2024-04-09": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gpt-4-1106-preview": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gpt-4-vision-preview": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gpt-4": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "gpt-4-0613": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "gpt-4-32k": {
            "token_limit": 32768,
            "supports_temperature": False
        },
        "gpt-4-32k-0613": {
            "token_limit": 32768,
            "supports_temperature": False
        },
        "gpt-4o": {
            "token_limit": 128000,
            "supports_temperature": True
        },
        "gpt-4o-mini": {
            "token_limit": 128000,
            "supports_temperature": True
        },
        "chatgpt-4o-latest": {
            "token_limit": 128000,
            "supports_temperature": True
        },
        "o1-preview": {
            "token_limit": 128000,
            "supports_temperature": True
        },
        "o1-mini": {
            "token_limit": 128000,
            "supports_temperature": True
        }
    },
    "google_genai": {
        "gemini-pro": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gemini-1.5-flash-latest": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gemini-1.5-pro-latest": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "models/embedding-001": {
            "token_limit": 2048,
            "supports_temperature": False
        }
    },
    "google_vertexai": {
        "gemini-1.5-flash": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gemini-1.5-pro": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gemini-1.0-pro": {
            "token_limit": 128000,
            "supports_temperature": False
        }
    },
    "ollama": {
        "command-r": {
            "token_limit": 12800,
            "supports_temperature": False
        },
        "codellama": {
            "token_limit": 16000,
            "supports_temperature": False
        },
        "dbrx": {
            "token_limit": 32768,
            "supports_temperature": False
        },
        "deepseek-coder:33b": {
            "token_limit": 16000,
            "supports_temperature": False
        },
        "falcon": {
            "token_limit": 2048,
            "supports_temperature": False
        },
        "llama2": {
            "token_limit": 4096,
            "supports_temperature": False
        },
        "llama2:7b": {
            "token_limit": 4096,
            "supports_temperature": False
        },
        "llama2:13b": {
            "token_limit": 4096,
            "supports_temperature": False
        },
        "llama2:70b": {
            "token_limit": 4096,
            "supports_temperature": False
        },
        "llama3": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "llama3:8b": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "llama3:70b": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "llama3.1": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "llama3.1:8b": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "llama3.1:70b": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "lama3.1:405b": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "llama3.2": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "llama3.2:1b": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "llama3.2:3b": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "llama3.3:70b": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "scrapegraph": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "mistral-small": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "mistral-openorca": {
            "token_limit": 32000,
            "supports_temperature": False
        },
        "mistral-large": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "grok-1": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "llava": {
            "token_limit": 4096,
            "supports_temperature": False
        },
        "mixtral:8x22b-instruct": {
            "token_limit": 65536,
            "supports_temperature": False
        },
        "nomic-embed-text": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "nous-hermes2:34b": {
            "token_limit": 4096,
            "supports_temperature": False
        },
        "orca-mini": {
            "token_limit": 2048,
            "supports_temperature": False
        },
        "phi3:3.8b": {
            "token_limit": 12800,
            "supports_temperature": False
        },
        "phi3:14b": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "qwen:0.5b": {
            "token_limit": 32000,
            "supports_temperature": False
        },
        "qwen:1.8b": {
            "token_limit": 32000,
            "supports_temperature": False
        },
        "qwen:4b": {
            "token_limit": 32000,
            "supports_temperature": False
        },
        "qwen:14b": {
            "token_limit": 32000,
            "supports_temperature": False
        },
        "qwen:32b": {
            "token_limit": 32000,
            "supports_temperature": False
        },
        "qwen:72b": {
            "token_limit": 32000,
            "supports_temperature": False
        },
        "qwen:110b": {
            "token_limit": 32000,
            "supports_temperature": False
        },
        "stablelm-zephyr": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "wizardlm2:8x22b": {
            "token_limit": 65536,
            "supports_temperature": False
        },
        "mistral": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gemma2": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gemma2:9b": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "gemma2:27b": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        # embedding models
        "shaw/dmeta-embedding-zh-small-q4": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "shaw/dmeta-embedding-zh-q4": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "chevalblanc/acge_text_embedding": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "martcreation/dmeta-embedding-zh": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "snowflake-arctic-embed": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "mxbai-embed-large": {
            "token_limit": 512,
            "supports_temperature": False
        }
    },
    "oneapi": {
        "qwen-turbo": {
            "token_limit": 6000,
            "supports_temperature": False
        }
    },
    "nvidia": {
        "meta/llama3-70b-instruct": {
            "token_limit": 419,
            "supports_temperature": False
        },
        "meta/llama3-8b-instruct": {
            "token_limit": 419,
            "supports_temperature": False
        },
        "nemotron-4-340b-instruct": {
            "token_limit": 1024,
            "supports_temperature": False
        },
        "databricks/dbrx-instruct": {
            "token_limit": 4096,
            "supports_temperature": False
        },
        "google/codegemma-7b": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "google/gemma-2b": {
            "token_limit": 2048,
            "supports_temperature": False
        },
        "google/gemma-7b": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "google/recurrentgemma-2b": {
            "token_limit": 2048,
            "supports_temperature": False
        },
        "meta/codellama-70b": {
            "token_limit": 16384,
            "supports_temperature": False
        },
        "meta/llama2-70b": {
            "token_limit": 4096,
            "supports_temperature": False
        },
        "microsoft/phi-3-mini-128k-instruct": {
            "token_limit": 122880,
            "supports_temperature": False
        },
        "mistralai/mistral-7b-instruct-v0.2": {
            "token_limit": 4096,
            "supports_temperature": False
        },
        "mistralai/mistral-large": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "mistralai/mixtral-8x22b-instruct-v0.1": {
            "token_limit": 32768,
            "supports_temperature": False
        },
        "mistralai/mixtral-8x7b-instruct-v0.1": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "snowflake/arctic": {
            "token_limit": 16384,
            "supports_temperature": False
        }
    },
    "groq": {
        "llama3-8b-8192": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "llama3-70b-8192": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "mixtral-8x7b-32768": {
            "token_limit": 32768,
            "supports_temperature": False
        },
        "gemma-7b-it": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "claude-3-haiku-20240307'": {
            "token_limit": 8192,
            "supports_temperature": False
        }
    },
    "toghetherai": {
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "mistralai/Mixtral-8x22B-Instruct-v0.1": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "stabilityai/stable-diffusion-xl-base-1.0": {
            "token_limit": 2048,
            "supports_temperature": False
        },
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "NousResearch/Hermes-3-Llama-3.1-405B-Turbo": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "Gryphe/MythoMax-L2-13b-Lite": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "Salesforce/Llama-Rank-V1": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "meta-llama/Meta-Llama-Guard-3-8B": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "meta-llama/Meta-Llama-3-70B-Instruct-Turbo": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "meta-llama/Llama-3-8b-chat-hf": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "meta-llama/Llama-3-70b-chat-hf": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "Qwen/Qwen2-72B-Instruct": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "google/gemma-2-27b-it": {
            "token_limit": 8192,
            "supports_temperature": False
        }
    },
    "anthropic": {
        "claude_instant": {
            "token_limit": 100000,
            "supports_temperature": False
        },
        "claude2": {
            "token_limit": 9000,
            "supports_temperature": False
        },
        "claude2.1": {
            "token_limit": 200000,
            "supports_temperature": False
        },
        "claude3": {
            "token_limit": 200000,
            "supports_temperature": False
        },
        "claude3.5": {
            "token_limit": 200000,
            "supports_temperature": False
        },
        "claude-3-opus-20240229": {
            "token_limit": 200000,
            "supports_temperature": False
        },
        "claude-3-sonnet-20240229": {
            "token_limit": 200000,
            "supports_temperature": False
        },
        "claude-3-haiku-20240307": {
            "token_limit": 200000,
            "supports_temperature": False
        },
        "claude-3-5-sonnet-20240620": {
            "token_limit": 200000,
            "supports_temperature": False
        },
        "claude-3-5-sonnet-20241022": {
            "token_limit": 200000,
            "supports_temperature": False
        },
        "claude-3-5-haiku-latest": {
            "token_limit": 200000,
            "supports_temperature": False
        }
    },
    "bedrock": {
        "anthropic.claude-3-haiku-20240307-v1:0": {
            "token_limit": 200000,
            "supports_temperature": False
        },
        "anthropic.claude-3-sonnet-20240229-v1:0": {
            "token_limit": 200000,
            "supports_temperature": False
        },
        "anthropic.claude-3-opus-20240229-v1:0": {
            "token_limit": 200000,
            "supports_temperature": False
        },
        "anthropic.claude-3-5-sonnet-20240620-v1:0": {
            "token_limit": 200000,
            "supports_temperature": False
        },
        "claude-3-5-haiku-latest": {
            "token_limit": 200000,
            "supports_temperature": False
        },
        "anthropic.claude-v2:1": {
            "token_limit": 200000,
            "supports_temperature": False
        },
        "anthropic.claude-v2": {
            "token_limit": 100000,
            "supports_temperature": False
        },
        "anthropic.claude-instant-v1": {
            "token_limit": 100000,
            "supports_temperature": False
        },
        "meta.llama3-8b-instruct-v1:0": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "meta.llama3-70b-instruct-v1:0": {
            "token_limit": 8192,
            "supports_temperature": False
        },
        "meta.llama2-13b-chat-v1": {
            "token_limit": 4096,
            "supports_temperature": False
        },
        "meta.llama2-70b-chat-v1": {
            "token_limit": 4096,
            "supports_temperature": False
        },
        "mistral.mistral-7b-instruct-v0:2": {
            "token_limit": 32768,
            "supports_temperature": False
        },
        "mistral.mixtral-8x7b-instruct-v0:1": {
            "token_limit": 32768,
            "supports_temperature": False
        },
        "mistral.mistral-large-2402-v1:0": {
            "token_limit": 32768,
            "supports_temperature": False
        },
        "mistral.mistral-small-2402-v1:0": {
            "token_limit": 32768,
            "supports_temperature": False
        },
        "amazon.titan-embed-text-v1": {
            "token_limit": 8000,
            "supports_temperature": False
        },
        "amazon.titan-embed-text-v2:0": {
            "token_limit": 8000,
            "supports_temperature": False
        },
        "cohere.embed-english-v3": {
            "token_limit": 512,
            "supports_temperature": False
        },
        "cohere.embed-multilingual-v3": {
            "token_limit": 512,
            "supports_temperature": False
        }
    },
    "mistralai": {
        "mistral-large-latest": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "open-mistral-nemo": {
            "token_limit": 128000,
            "supports_temperature": False
        },
        "codestral-latest": {
            "token_limit": 32000,
            "supports_temperature": False
        }
    },
    "togetherai": {
        "Meta-Llama-3.1-70B-Instruct-Turbo": {
            "token_limit": 128000,
            "supports_temperature": False
        }
    }
}
