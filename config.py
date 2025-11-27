"""
Configuration file for API keys and settings.
You can set your API keys here directly, or use environment variables.
"""

# ============================================
# API KEYS CONFIGURATION
# ============================================
# Option 1: Set API keys directly here (not recommended for production)
# Option 2: Use environment variables (recommended)
# Option 3: Enter keys in the Streamlit app sidebar

# Voyage AI API Key (for embeddings)
# Get your key at: https://www.voyageai.com/
VOYAGE_API_KEY = 'pa-bYsH0zeM9TAAA_ZVFcx_IDeBaE1VAO_Biz8cjEtAx6F'  # Set your key here, or use environment variable VOYAGE_API_KEY

# OpenRouter API Key (for suggestions - requires credits)
# Get your key at: https://openrouter.ai/keys
OPENROUTER_API_KEY = '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'  # Set your key here if you have credits, or use environment variable OPENROUTER_API_KEY

# Hugging Face API Key (for suggestions - FREE tier available!)
# Get your free key at: https://huggingface.co/settings/tokens
# Free tier includes access to many models
HF_API_KEY = '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'  # Set your key here, or use environment variable HF_API_KEY

# Together AI API Key (for suggestions - FREE tier available!)
# Get your free key at: https://api.together.xyz/settings/api-keys
# Free tier: $25 free credits, no credit card needed
TOGETHER_API_KEY = 'edaa7a6c2be40a49f41cbd6bea8b8ebd7770a80ee5d658aa7a11e9adb0ac75ad'  # Set your key here, or use environment variable TOGETHER_API_KEY

# ============================================
# API ENDPOINTS (usually don't need to change)
# ============================================
VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
HF_API_URL = "https://api-inference.huggingface.co"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# ============================================
# MODEL NAMES (usually don't need to change)
# ============================================
VOYAGE_MODEL = "voyage-lite-02-instruct"
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # FREE on Hugging Face!
TOGETHER_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # FREE on Together AI (serverless)!
