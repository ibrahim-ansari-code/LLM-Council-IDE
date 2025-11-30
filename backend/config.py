"""
Configuration for the LLM Council backend.
"""
import os
from typing import List

# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Council Models - these are the LLMs that will provide initial responses
# Based on OpenRouter rankings, using top models for coding:
# - Claude Sonnet 4.5 (amazing for coding)
# - Gemini 2.5 Flash (fast and capable)
# - GPT-4o (reliable)
COUNCIL_MODELS = [
    "anthropic/claude-sonnet-4.5",  # Claude Sonnet 4.5 - excellent for coding
    "google/gemini-2.5-flash",      # Gemini 2.5 Flash
    "openai/gpt-4o",                 # GPT-4o
    "openai/gpt-3.5-turbo",         # Fallback option
]

# Chairman Model - this LLM will synthesize the final response
CHAIRMAN_MODEL = "anthropic/claude-sonnet-4.5"  # Using Claude Sonnet 4.5 as chairman (best for coding)

# Token Limits - adjust based on your OpenRouter credits
# Lower values = less credits needed, but shorter responses
# Increase this when you add more credits to your account
# For code generation, we need at least 5000 tokens to get complete code blocks and multiple files
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "5000"))  # Default: 5000 (needed for complete code and multi-file projects)

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# CORS Configuration
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

