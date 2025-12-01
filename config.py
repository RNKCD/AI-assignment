"""
Configuration file for API keys and settings.
Loads API keys from .env file or environment variables for security.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================
# API KEYS CONFIGURATION
# ============================================
# SECURITY: API keys are loaded from .env file (not committed to git)
# Option 1: Use .env file (recommended for local development)
# Option 2: Use environment variables (recommended for production)
# Option 3: Enter keys in the Streamlit app sidebar

# Voyage AI API Key (for embeddings)
# Get your key at: https://www.voyageai.com/
VOYAGE_API_KEY = os.getenv('VOYAGE_API_KEY', None)

# OpenRouter API Key (for suggestions - requires credits)
# Get your key at: https://openrouter.ai/keys
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', None)

# Hugging Face API Key (for suggestions - FREE tier available!)
# Get your free key at: https://huggingface.co/settings/tokens
# Free tier includes access to many models
HF_API_KEY = os.getenv('HF_API_KEY', None)

# Together AI API Key (for suggestions - FREE tier available!)
# Get your free key at: https://api.together.xyz/settings/api-keys
# Free tier: $25 free credits, no credit card needed
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY', None)

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
